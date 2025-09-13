from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLM(ABC):
    @abstractmethod
    def chat_json(
        self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024
    ) -> str:
        """Return STRICT JSON string (not Python dict)."""
        ...
