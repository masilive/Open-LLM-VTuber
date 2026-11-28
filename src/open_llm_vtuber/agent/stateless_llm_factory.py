from typing import Type

from loguru import logger

from .stateless_llm.stateless_llm_interface import StatelessLLMInterface
from .stateless_llm.stateless_llm_with_template import (
    AsyncLLMWithTemplate as StatelessLLMWithTemplate,
)
from .stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleLLM
from .stateless_llm.gemini_llm import AsyncLLM as GeminiLLM
from .stateless_llm.ollama_llm import OllamaLLM
from .stateless_llm.claude_llm import AsyncLLM as ClaudeLLM
from .api_key_resolver import resolve_api_key


class LLMFactory:
    @staticmethod
    def create_llm(llm_provider, **kwargs) -> Type[StatelessLLMInterface]:
        """Create an LLM based on the configuration.

        Args:
            llm_provider: The type of LLM to create
            **kwargs: Additional arguments
        """
        logger.info(f"Initializing LLM: {llm_provider}")

        env_key_names = {
            "openai_llm": ("OPENAI_API_KEY",),
            "claude_llm": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
            "windsurf_llm": ("WINDSURF_API_KEY", "CODEIUM_API_KEY"),
            "gemini_llm": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        }.get(llm_provider)
        if env_key_names:
            kwargs["llm_api_key"] = resolve_api_key(
                kwargs.get("llm_api_key"), env_key_names
            )

        if llm_provider == "gemini_llm":
            return GeminiLLM(
                model=kwargs.get("model"),
                llm_api_key=kwargs.get("llm_api_key"),
                temperature=kwargs.get("temperature"),
                thinking_level=kwargs.get("thinking_level"),
            )

        if (
            llm_provider == "openai_compatible_llm"
            or llm_provider == "openai_llm"
            or llm_provider == "zhipu_llm"
            or llm_provider == "deepseek_llm"
            or llm_provider == "groq_llm"
            or llm_provider == "mistral_llm"
            or llm_provider == "lmstudio_llm"
            or llm_provider == "windsurf_llm"
        ):
            return OpenAICompatibleLLM(
                model=kwargs.get("model"),
                base_url=kwargs.get("base_url"),
                llm_api_key=kwargs.get("llm_api_key"),
                organization_id=kwargs.get("organization_id"),
                project_id=kwargs.get("project_id"),
                temperature=kwargs.get("temperature"),
            )
        if llm_provider == "stateless_llm_with_template":
            return StatelessLLMWithTemplate(
                model=kwargs.get("model"),
                base_url=kwargs.get("base_url"),
                llm_api_key=kwargs.get("llm_api_key"),
                organization_id=kwargs.get("organization_id"),
                template=kwargs.get("template"),
                project_id=kwargs.get("project_id"),
            )
        if llm_provider == "ollama_llm":
            return OllamaLLM(
                model=kwargs.get("model"),
                base_url=kwargs.get("base_url"),
                llm_api_key=kwargs.get("llm_api_key"),
                organization_id=kwargs.get("organization_id"),
                project_id=kwargs.get("project_id"),
                temperature=kwargs.get("temperature"),
                keep_alive=kwargs.get("keep_alive"),
                unload_at_exit=kwargs.get("unload_at_exit"),
            )

        elif llm_provider == "llama_cpp_llm":
            from .stateless_llm.llama_cpp_llm import LLM as LlamaLLM

            return LlamaLLM(
                model_path=kwargs.get("model_path"),
            )
        elif llm_provider == "claude_llm":
            return ClaudeLLM(
                system=kwargs.get("system_prompt"),
                base_url=kwargs.get("base_url"),
                model=kwargs.get("model"),
                llm_api_key=kwargs.get("llm_api_key"),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")


# Creating an LLM instance using a factory
# llm_instance = LLMFactory.create_llm("ollama", **config_dict)
