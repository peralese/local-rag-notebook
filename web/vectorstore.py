# vectorstore.py
import os, chromadb
from chromadb.config import Settings
from embeddings import EmbeddingFunction

class VectorStore:
    def __init__(self, persist_dir: str | None = None):
        persist_dir = persist_dir or os.getenv("VECTOR_DIR", "/data/chroma")
        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        self.embedding_fn = EmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self):
        try:
            self.client.delete_collection("docs")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, documents, metadatas=None):
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def count(self):
        return self.collection.count()

    def query(self, text, k=5):
        return self.collection.query(query_texts=[text], n_results=k)
