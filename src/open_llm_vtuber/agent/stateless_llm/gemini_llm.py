"""Native Google Gemini streaming LLM implementation."""

import base64
import binascii
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from loguru import logger

from ...mcpp.types import ToolCallFunctionObject, ToolCallObject
from .stateless_llm_interface import StatelessLLMInterface

_DATA_URL_PATTERN = re.compile(
    r"^data:(?P<mime_type>[^;,]+);base64,(?P<data>.+)$", re.DOTALL
)


class AsyncLLM(StatelessLLMInterface):
    """Stateless streaming Gemini client using the native google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        llm_api_key: str | None = None,
        temperature: float = 1.0,
        thinking_level: str = "medium",
    ) -> None:
        """Initialize a native Gemini streaming client.

        Args:
            model: Gemini model identifier. The default tracks the latest Flash model.
            llm_api_key: Gemini Developer API key. If omitted, the SDK can use its
                own supported environment configuration.
            temperature: Sampling temperature.
            thinking_level: Reasoning depth: minimal, low, medium, or high.
        """
        self.model = model
        self.temperature = temperature
        self.thinking_level = self._parse_thinking_level(thinking_level)
        self.client = genai.Client(
            api_key=llm_api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )

        logger.info(
            "Initialized native Gemini LLM with model {} and thinking level {}",
            self.model,
            self.thinking_level.value,
        )

    @staticmethod
    def _parse_thinking_level(thinking_level: str) -> types.ThinkingLevel:
        """Parse a user-facing thinking level into the SDK enum."""
        normalized = str(thinking_level or "medium").strip().upper()
        try:
            return types.ThinkingLevel[normalized]
        except KeyError as exc:
            valid_levels = ", ".join(
                level.name.lower()
                for level in types.ThinkingLevel
                if level != types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED
            )
            raise ValueError(
                f"Unsupported Gemini thinking level '{thinking_level}'. "
                f"Expected one of: {valid_levels}."
            ) from exc

    @staticmethod
    def _flatten_text_content(content: Any) -> str:
        """Flatten OpenAI-style text content into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_items = []
            for item in content:
                if isinstance(item, str):
                    text_items.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_items.append(str(item.get("text", "")))
            return "\n".join(item for item in text_items if item)
        return str(content)

    @staticmethod
    def _part_from_data_url(data_url: str) -> types.Part | None:
        """Convert a base64 data URL into a Gemini inline-data part."""
        match = _DATA_URL_PATTERN.match(data_url)
        if not match:
            logger.warning(
                "Unsupported Gemini image input format; expected a base64 data URL."
            )
            return None
        try:
            image_bytes = base64.b64decode(match.group("data"), validate=True)
        except (binascii.Error, ValueError) as exc:
            logger.warning(f"Skipping invalid base64 Gemini image input: {exc}")
            return None
        return types.Part.from_bytes(
            data=image_bytes,
            mime_type=match.group("mime_type"),
        )

    def _parts_from_content(self, content: Any) -> list[types.Part]:
        """Convert OpenAI-style message content into Gemini content parts."""
        parts: list[types.Part] = []
        if isinstance(content, str):
            if content:
                parts.append(types.Part.from_text(text=content))
            return parts
        if not isinstance(content, list):
            text = self._flatten_text_content(content)
            if text:
                parts.append(types.Part.from_text(text=text))
            return parts

        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(types.Part.from_text(text=item))
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = str(item.get("text", ""))
                if text:
                    parts.append(types.Part.from_text(text=text))
            elif item_type == "image_url":
                image_url = item.get("image_url") or {}
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if isinstance(url, str):
                    image_part = self._part_from_data_url(url)
                    if image_part is not None:
                        parts.append(image_part)
        return parts

    @staticmethod
    def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
        """Parse OpenAI-style function arguments into a dictionary."""
        if isinstance(raw_arguments, dict):
            return raw_arguments
        try:
            parsed = json.loads(raw_arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            logger.warning(
                "Invalid tool-call arguments in conversation history; using empty arguments."
            )
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    @staticmethod
    def _tool_names_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
        """Build a tool-call ID to function-name lookup from OpenAI-style history."""
        names_by_id: dict[str, str] = {}
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                tool_call_id = tool_call.get("id")
                function_name = function.get("name")
                if tool_call_id and function_name:
                    names_by_id[str(tool_call_id)] = str(function_name)
        return names_by_id

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> tuple[list[types.Content], str | None]:
        """Convert OpenAI-style conversation messages into Gemini contents."""
        system_parts = [system] if system else []
        contents: list[types.Content] = []
        tool_names_by_id = self._tool_names_by_id(messages)

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role == "system":
                system_text = self._flatten_text_content(content)
                if system_text:
                    system_parts.append(system_text)
                continue

            if role == "assistant":
                parts = self._parts_from_content(content)
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    function_name = function.get("name")
                    if not function_name:
                        continue
                    parts.append(
                        types.Part.from_function_call(
                            name=str(function_name),
                            args=self._parse_tool_arguments(function.get("arguments")),
                        )
                    )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            if role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "")
                function_name = tool_names_by_id.get(
                    tool_call_id, tool_call_id or "tool_result"
                )
                result_text = self._flatten_text_content(content)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=function_name,
                                response={"result": result_text},
                            )
                        ],
                    )
                )
                continue

            if role == "user":
                parts = self._parts_from_content(content)
                if parts:
                    contents.append(types.Content(role="user", parts=parts))
                continue

            logger.warning(f"Skipping unsupported Gemini message role: {role}")

        system_instruction = "\n\n".join(part for part in system_parts if part)
        return contents, system_instruction or None

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[types.Tool] | None:
        """Convert OpenAI-style function tools into Gemini function declarations."""
        declarations: list[types.FunctionDeclaration] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            declarations.append(
                types.FunctionDeclaration(
                    name=str(name),
                    description=str(function.get("description") or ""),
                    parameters_json_schema=function.get("parameters")
                    or {"type": "object", "properties": {}},
                )
            )
        if not declarations:
            return None
        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def _convert_function_call(
        function_call: types.FunctionCall,
        index: int,
    ) -> ToolCallObject:
        """Convert a Gemini function call into the project's OpenAI-style type."""
        arguments = function_call.args or {}
        return ToolCallObject(
            id=function_call.id or f"gemini-function-call-{index}",
            type="function",
            index=index,
            function=ToolCallFunctionObject(
                name=function_call.name or "",
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    async def close(self) -> None:
        """Close asynchronous network resources owned by the Gemini client."""
        aclose = getattr(self.client.aio, "aclose", None)
        if aclose is not None:
            await aclose()

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str | list[ToolCallObject]]:
        """Generate a streaming Gemini chat completion.

        Args:
            messages: OpenAI-style conversation messages.
            system: System prompt for this completion.
            tools: OpenAI-style function tool definitions.

        Yields:
            Incremental text chunks followed by a list of complete tool calls when
            Gemini requests function execution.
        """
        stream = None
        try:
            contents, system_instruction = self._convert_messages(messages, system)
            generation_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=self.temperature,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=False,
                    thinking_level=self.thinking_level,
                ),
                tools=self._convert_tools(tools),
            )

            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generation_config,
            )

            function_calls: list[types.FunctionCall] = []
            async for chunk in stream:
                if chunk.function_calls:
                    function_calls.extend(chunk.function_calls)
                if chunk.text:
                    yield chunk.text

            if function_calls:
                yield [
                    self._convert_function_call(function_call, index)
                    for index, function_call in enumerate(function_calls)
                ]

        except genai_errors.APIError as exc:
            logger.error(f"Gemini API error while generating a response: {exc}")
            yield "Error calling the Gemini endpoint: Error occurred while generating response. See the logs for details."
        except Exception as exc:
            logger.exception(
                f"Unexpected Gemini error while generating a response: {exc}"
            )
            yield "Error calling the Gemini endpoint: Unexpected error while generating response. See the logs for details."
        finally:
            if stream is not None:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()
