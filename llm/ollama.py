from typing import Dict, List

import requests

from .base import LLM


class OllamaLLM(LLM):
    def __init__(self, model: str = "llama3.1:8b", endpoint: str = "http://localhost:11434"):
        self.model = model
        self.url = endpoint.rstrip("/") + "/api/chat"

    def chat_json(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "format": "json",     # forces models to return JSON
            "stream": False,       
        }
        r = requests.post(self.url, json=payload, timeout=(10, 600))
        r.raise_for_status()
        # Ollama returns {"message":{"content":"<JSON string>"}}
        content = r.json()["message"]["content"].strip()
        return content
