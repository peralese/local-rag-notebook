from .base import LLM
from .ollama import OllamaLLM


def make_llm(
    backend: str = "ollama",
    model: str = "llama3.1:8b",
    endpoint: str = "http://localhost:11434",
    offline: bool = True,
) -> LLM:
    backend = (backend or "ollama").lower()

    # Offline guard: only allow localhost endpoints
    if offline and not (
        endpoint.startswith("http://localhost") or endpoint.startswith("http://127.0.0.1")
    ):
        raise RuntimeError("Offline mode: only localhost endpoints are allowed.")

    if backend == "ollama":
        return OllamaLLM(model=model, endpoint=endpoint)

    raise RuntimeError(f"Unsupported backend: {backend}")
