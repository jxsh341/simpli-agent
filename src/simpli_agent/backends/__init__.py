"""Backend implementations for simpli-agent."""

from .base import Backend
from .hermes import HermesBackend

try:
    from .openai import OpenAIBackend
except ImportError:
    OpenAIBackend = None  # type: ignore

try:
    from .anthropic import AnthropicBackend
except ImportError:
    AnthropicBackend = None  # type: ignore

try:
    from .ollama import OllamaBackend
except ImportError:
    OllamaBackend = None  # type: ignore

__all__ = ["Backend", "HermesBackend"]
if OpenAIBackend:
    __all__.append("OpenAIBackend")
if AnthropicBackend:
    __all__.append("AnthropicBackend")
if OllamaBackend:
    __all__.append("OllamaBackend")
