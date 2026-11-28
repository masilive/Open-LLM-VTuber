"""Focused tests for the native Gemini LLM integration."""

import json
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src.open_llm_vtuber.agent import stateless_llm_factory
from src.open_llm_vtuber.agent.stateless_llm.gemini_llm import AsyncLLM as GeminiLLM
from src.open_llm_vtuber.config_manager.stateless_llm import GeminiConfig
from src.open_llm_vtuber.mcpp.types import ToolCallObject


class FakeGeminiModels:
    """Fake async model endpoint matching the google-genai streaming contract."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def generate_content_stream(self, *, model, contents, config):
        """Record a streaming request and return an async chunk iterator."""
        self.calls.append({"model": model, "contents": contents, "config": config})

        async def iterate_chunks():
            for chunk in self.chunks:
                yield chunk

        return iterate_chunks()


def install_fake_gemini_client(llm: GeminiLLM, chunks) -> FakeGeminiModels:
    """Replace a Gemini client's async model endpoint with a fake."""
    models = FakeGeminiModels(chunks)
    llm.client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return models


def text_chunk(text: str) -> genai_types.GenerateContentResponse:
    """Build a Gemini response chunk containing one text part."""
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part.from_text(text=text)]
                )
            )
        ]
    )


def test_gemini_config_defaults_to_latest_flash_medium_thinking() -> None:
    """Gemini configuration defaults match the requested deployment shape."""
    config = GeminiConfig()

    assert config.llm_api_key == "${GEMINI_API_KEY}"
    assert config.model == "gemini-flash-latest"
    assert config.thinking_level == "medium"
    assert config.temperature == 1.0


def test_gemini_rejects_unknown_thinking_level() -> None:
    """Invalid Gemini thinking levels fail fast during construction."""
    with pytest.raises(ValueError, match="Unsupported Gemini thinking level"):
        GeminiLLM(llm_api_key="test-key", thinking_level="extreme")


@pytest.mark.asyncio
async def test_gemini_streams_text_with_medium_thinking_config() -> None:
    """Native Gemini streaming uses medium thinking and yields incremental text."""
    llm = GeminiLLM(
        model="gemini-flash-latest",
        llm_api_key="test-key",
        temperature=0.4,
        thinking_level="medium",
    )
    models = install_fake_gemini_client(
        llm, [text_chunk("Hello"), text_chunk(" world")]
    )

    events = [
        event
        async for event in llm.chat_completion(
            [{"role": "user", "content": "Say hello"}], system="Be concise"
        )
    ]

    assert events == ["Hello", " world"]
    assert len(models.calls) == 1
    call = models.calls[0]
    assert call["model"] == "gemini-flash-latest"
    assert call["config"].temperature == 0.4
    assert call["config"].system_instruction == "Be concise"
    assert (
        call["config"].thinking_config.thinking_level
        == genai_types.ThinkingLevel.MEDIUM
    )
    assert call["config"].thinking_config.include_thoughts is False
    assert call["contents"][0].role == "user"
    assert call["contents"][0].parts[0].text == "Say hello"


@pytest.mark.asyncio
async def test_gemini_function_calls_convert_to_openai_shape() -> None:
    """Gemini function calls are exposed through the project's tool-call contract."""
    function_chunk = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part.from_function_call(
                            name="search", args={"query": "weather"}
                        )
                    ],
                )
            )
        ]
    )
    llm = GeminiLLM(llm_api_key="test-key")
    models = install_fake_gemini_client(llm, [function_chunk])

    events = [
        event
        async for event in llm.chat_completion(
            [{"role": "user", "content": "What is the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search information",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        )
    ]

    assert len(events) == 1
    assert len(events[0]) == 1
    tool_call = events[0][0]
    assert isinstance(tool_call, ToolCallObject)
    assert tool_call.type == "function"
    assert tool_call.id == "gemini-function-call-0"
    assert tool_call.function.name == "search"
    assert json.loads(tool_call.function.arguments) == {"query": "weather"}

    declarations = models.calls[0]["config"].tools[0].function_declarations
    assert declarations[0].name == "search"
    assert declarations[0].parameters_json_schema["required"] == ["query"]


def test_gemini_converts_text_image_and_system_messages() -> None:
    """OpenAI-style multimodal and system messages convert into Gemini contents."""
    llm = GeminiLLM(llm_api_key="test-key")
    contents, system_instruction = llm._convert_messages(
        [
            {"role": "system", "content": "Remember interruptions."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGk="},
                    },
                ],
            },
        ],
        system="You are helpful.",
    )

    assert system_instruction == "You are helpful.\n\nRemember interruptions."
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "What is in this image?"
    assert contents[0].parts[1].inline_data.mime_type == "image/png"
    assert contents[0].parts[1].inline_data.data == b"hi"


def test_gemini_converts_openai_tool_history() -> None:
    """OpenAI-style assistant calls and tool results become Gemini function parts."""
    llm = GeminiLLM(llm_api_key="test-key")
    contents, _ = llm._convert_messages(
        [
            {"role": "user", "content": "Search for weather."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "weather"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        ],
        system=None,
    )

    assert contents[1].role == "model"
    assert contents[1].parts[0].function_call.name == "search"
    assert contents[1].parts[0].function_call.args == {"query": "weather"}
    assert contents[2].role == "user"
    assert contents[2].parts[0].function_response.name == "search"
    assert contents[2].parts[0].function_response.response == {"result": "sunny"}


def test_gemini_content_conversion_edge_cases() -> None:
    """Malformed or mixed content is normalized without raising."""
    llm = GeminiLLM(llm_api_key="test-key")

    assert llm._flatten_text_content(None) == ""
    assert llm._flatten_text_content("text") == "text"
    assert llm._flatten_text_content(["a", {"type": "text", "text": "b"}, 4]) == "a\nb"
    assert llm._flatten_text_content(4) == "4"
    assert llm._part_from_data_url("https://example.invalid/image.png") is None
    assert llm._part_from_data_url("data:image/png;base64,!!!") is None

    numeric_parts = llm._parts_from_content(4)
    assert numeric_parts[0].text == "4"
    assert llm._parts_from_content([""]) == []
    mixed_parts = llm._parts_from_content(
        [
            "first",
            4,
            {"type": "text", "text": ""},
            {"type": "unknown", "text": "ignored"},
            {"type": "image_url", "image_url": {}},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/x.png"},
            },
            {"type": "image_url", "image_url": "data:image/png;base64,aGk="},
        ]
    )
    assert mixed_parts[0].text == "first"
    assert mixed_parts[1].inline_data.data == b"hi"


def test_gemini_tool_argument_and_schema_edge_cases() -> None:
    """Tool argument and declaration conversion handles incomplete input."""
    llm = GeminiLLM(llm_api_key="test-key")

    assert llm._parse_tool_arguments({"already": "dict"}) == {"already": "dict"}
    assert llm._parse_tool_arguments('["not-object"]') == {"value": ["not-object"]}
    assert llm._parse_tool_arguments("not-json") == {}

    assert llm._convert_tools([None, {}]) is None
    converted = llm._convert_tools(
        [None, {}, {"type": "function", "function": {"name": "lookup"}}]
    )
    declaration = converted[0].function_declarations[0]
    assert declaration.name == "lookup"
    assert declaration.description == ""
    assert declaration.parameters_json_schema == {"type": "object", "properties": {}}


def test_gemini_history_conversion_edge_cases() -> None:
    """Incomplete tool history and unknown roles are skipped or safely mapped."""
    llm = GeminiLLM(llm_api_key="test-key")
    contents, system_instruction = llm._convert_messages(
        [
            {"role": "system", "content": ""},
            {
                "role": "system",
                "content": ["system", {"type": "text", "text": "rules"}],
            },
            {"role": "assistant", "content": ""},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    "not-a-tool-call",
                    {"id": "missing-name", "function": {}},
                    {"id": "call-1", "function": {"name": "lookup", "arguments": 4}},
                ],
            },
            {"role": "tool", "tool_call_id": "unknown-call", "content": None},
            {"role": "user", "content": ""},
            {"role": "unexpected", "content": "ignored"},
        ],
        system=None,
    )

    assert system_instruction == "system\nrules"
    assert len(contents) == 2
    assert contents[0].role == "model"
    assert contents[0].parts[0].function_call.name == "lookup"
    assert contents[0].parts[0].function_call.args == {}
    assert contents[1].role == "user"
    assert contents[1].parts[0].function_response.name == "unknown-call"
    assert contents[1].parts[0].function_response.response == {"result": ""}


@pytest.mark.asyncio
async def test_gemini_stream_finishes_when_iterator_has_no_close_hook() -> None:
    """Stream cleanup tolerates async iterators without an aclose method."""

    class EmptyAsyncIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class EmptyModels:
        async def generate_content_stream(self, *, model, contents, config):
            return EmptyAsyncIterator()

    llm = GeminiLLM(llm_api_key="test-key")
    llm.client = SimpleNamespace(aio=SimpleNamespace(models=EmptyModels()))

    events = [
        event
        async for event in llm.chat_completion([{"role": "user", "content": "hello"}])
    ]

    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (
            genai_errors.APIError(500, {"error": {"message": "backend"}}),
            "Error calling the Gemini endpoint: Error occurred",
        ),
        (
            RuntimeError("unexpected"),
            "Error calling the Gemini endpoint: Unexpected error",
        ),
    ],
)
async def test_gemini_stream_errors_return_user_safe_text(raised, expected) -> None:
    """Gemini API and unexpected failures are converted into user-safe text."""

    class FailingModels:
        async def generate_content_stream(self, *, model, contents, config):
            raise raised

    llm = GeminiLLM(llm_api_key="test-key")
    llm.client = SimpleNamespace(aio=SimpleNamespace(models=FailingModels()))

    events = [
        event
        async for event in llm.chat_completion([{"role": "user", "content": "hello"}])
    ]

    assert len(events) == 1
    assert events[0].startswith(expected)


@pytest.mark.asyncio
async def test_gemini_close_delegates_to_async_client() -> None:
    """Closing the Gemini provider closes SDK asynchronous network resources."""
    llm = GeminiLLM(llm_api_key="test-key")
    closed = False

    async def aclose():
        nonlocal closed
        closed = True

    llm.client = SimpleNamespace(aio=SimpleNamespace(aclose=aclose))
    await llm.close()

    assert closed is True


@pytest.mark.asyncio
async def test_gemini_close_is_safe_without_async_close_hook() -> None:
    """Closing remains safe when a test or custom client has no close hook."""
    llm = GeminiLLM(llm_api_key="test-key")
    llm.client = SimpleNamespace(aio=SimpleNamespace())
    await llm.close()


@pytest.mark.asyncio
async def test_basic_memory_agent_close_delegates_to_llm() -> None:
    """Service-context cleanup can reach provider resources through the agent."""
    from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent

    class ClosableLLM:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    llm = ClosableLLM()
    agent = BasicMemoryAgent(llm=llm, system="system", live2d_model=None)
    await agent.close()

    assert llm.closed is True


def test_factory_resolves_gemini_environment_key(monkeypatch) -> None:
    """LLMFactory resolves GEMINI_API_KEY before constructing the native client."""
    captured = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key")
    monkeypatch.setattr(stateless_llm_factory, "GeminiLLM", FakeGemini)

    stateless_llm_factory.LLMFactory.create_llm(
        "gemini_llm",
        model="gemini-flash-latest",
        llm_api_key="${GEMINI_API_KEY}",
        temperature=1.0,
        thinking_level="medium",
    )

    assert captured["llm_api_key"] == "gemini-env-key"
    assert captured["model"] == "gemini-flash-latest"
    assert captured["thinking_level"] == "medium"
