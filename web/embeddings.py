# embeddings.py
import os

BACKEND = os.getenv("EMBED_BACKEND", "fastembed")  # fastembed | ollama

if BACKEND == "ollama":
    import requests
    class OllamaEmbeddingFunction:
        def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 300):
            self.model = model or os.getenv("EMBED_MODEL", "nomic-embed-text")
            self.host = (host or os.getenv("OLLAMA_HOST", "http://ollama:11434")).rstrip("/")
            self.timeout = timeout

        def __call__(self, input: list[str]) -> list[list[float]]:
            out = []
            for text in input:
                r = requests.post(f"{self.host}/api/embeddings",
                                  json={"model": self.model, "input": text},
                                  timeout=self.timeout)
                r.raise_for_status()
                data = r.json() or {}
                if isinstance(data.get("embedding"), list) and data["embedding"]:
                    out.append(data["embedding"])
                elif isinstance(data.get("embeddings"), list) and data["embeddings"]:
                    out.append(data["embeddings"][0])
                else:
                    out.append([0.0]*384)  # safe fallback
            return out

    EmbeddingFunction = OllamaEmbeddingFunction

else:
    from fastembed import TextEmbedding
    class FastEmbedFunction:
        def __init__(self, model: str | None = None):
            self.model_name = model or os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
            self.model = TextEmbedding(model_name=self.model_name)

        def __call__(self, input: list[str]) -> list[list[float]]:
            return [vec for vec in self.model.embed(input)]

    EmbeddingFunction = FastEmbedFunction
