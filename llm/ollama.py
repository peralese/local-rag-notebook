# llm/ollama.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests

DEFAULT_OLLAMA = "http://localhost:11434"
# (connect timeout, read timeout) – generous read to allow initial model load
TIMEOUT = (10.0, 120.0)
DEFAULT_OLLAMA = "http://localhost:11434"

def _timeouts() -> tuple[float, float]:
    """Return (connect_timeout, read_timeout) in seconds; env-overridable."""
    # Defaults: 10s connect, 600s read (10 minutes) to handle long CPU generations
    ct = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10"))
    rt = float(os.getenv("OLLAMA_READ_TIMEOUT", "600"))
    return (ct, rt)


def _normalize_endpoint(ep: Optional[str]) -> str:
    """--endpoint > OLLAMA_HOST > default; ensure scheme; strip trailing slash."""
    cand = (ep or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA).strip()
    if not re.match(r"^https?://", cand):
        cand = "http://" + cand
    return cand.rstrip("/")


class OllamaLLM:
    """
    Minimal Ollama client used by synthesizer via llm.factory.make_llm().

    Typical construction:
        llm = OllamaLLM(model="llama3.1:8b", endpoint="http://localhost:11435", keep_alive="2h")

    The synthesizer calls:
        llm.chat_json(messages, temperature=..., max_tokens=...)
    We also provide `generate(...)` for callers that use /api/generate.
    """

    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        keep_alive: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.base = _normalize_endpoint(endpoint)
        self.keep_alive = keep_alive

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **_: Any,
    ) -> str:
        """
        Call /api/chat once (non-streaming) and return assistant text content.
        The synthesizer expects a JSON string response; we return the raw text,
        which should be JSON per the prompt template.
        """
        url = f"{self.base}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_tokens is not None:
            # Ollama uses num_predict for token limit
            options["num_predict"] = int(max_tokens)
        if options:
            payload["options"] = options

        r = requests.post(url, json=payload, timeout=_timeouts())
        r.raise_for_status()
        data = r.json()
        # Common shapes:
        #  - {"message":{"role":"assistant","content":"..."}}
        #  - {"response":"..."} (older/alt shape)
        msg = data.get("message", {})
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"]
        return data.get("response", "")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **_: Any,
    ) -> str:
        """
        Call /api/generate once (non-streaming) and return the 'response' text.
        Provided for back-compat with any callers not using chat_json.
        """
        if stream:
            stream = False
        if system:
            # Simple system prefix; align with your prompt template if needed.
            prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"

        url = f"{self.base}/api/generate"
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        if options:
            payload["options"] = options

        r = requests.post(url, json=payload, timeout=_timeouts())
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    # Convenience alias some factories call
    __call__ = generate

