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
        # upsert prevents duplicates by id (content-hash)
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def count(self):
        return self.collection.count()

    # ---- quick wins ----
    def delete_by_path(self, path: str, exact: bool = False) -> dict:
        if exact:
            where = {"path": {"$eq": path}}
            before = self._count_where(where)
            self.collection.delete(where=where)
            after = self._count_where(where)
            return {"deleted": max(0, before - after), "remaining_matching": after}

        # substring fallback: fetch metadatas, filter ids, then delete by ids
        res = self.collection.get(include=["metadatas"], limit=100000)
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        to_delete = []
        for i, m in zip(ids, metas):
            mpath = (m or {}).get("path") or ""
            if path in mpath:
                to_delete.append(i)
        if to_delete:
            self.collection.delete(ids=to_delete)
        return {"deleted": len(to_delete), "remaining_matching": 0}


    def _count_where(self, where: dict | None) -> int:
        try:
            res = self.collection.get(where=where, limit=100000)
            ids = res.get("ids", []) or []
            return len(ids)
        except Exception:
            return 0

    def query(self, text, k=5, include=None):
        # Backward-compatible simple query
        allowed = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}
        include = include or ["documents", "metadatas", "distances"]
        include = [x for x in include if x in allowed]
        return self.collection.query(query_texts=[text], n_results=k, include=include)

    def query_advanced(self, text: str, k: int = 5, include=None, where: dict | None = None,
                       distance_threshold: float | None = None, add_scores: bool = True) -> dict:
        """Query with optional metadata filters and distance threshold.
        Adds normalized scores (1 - distance) if add_scores is True.
        """
        allowed = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}
        include = include or ["documents", "metadatas", "distances"]
        include = [x for x in include if x in allowed]

        res = self.collection.query(query_texts=[text], n_results=k, include=include, where=where)

        # Compute normalized scores from distances (cosine space ⇒ score = 1 - distance)
        if add_scores:
            dists = res.get("distances") or []
            scores = []
            for row in dists:
                if row is None:
                    scores.append([])
                else:
                    scores.append([(1 - d) if d is not None else None for d in row])
            res["scores"] = scores

        # Apply distance threshold filter if provided
        if distance_threshold is not None:
            mask_rows = []
            dists = res.get("distances") or []
            for row in dists:
                mask = [(d is not None and d <= distance_threshold) for d in row] if row else []
                mask_rows.append(mask)

            def apply_mask(key):
                rows = res.get(key)
                if not rows:
                    return
                masked_rows = []
                for row, mask in zip(rows, mask_rows):
                    if row is None:
                        masked_rows.append(row)
                        continue
                    masked_rows.append([v for v, keep in zip(row, mask) if keep])
                res[key] = masked_rows

            for key in ["documents", "metadatas", "distances", "embeddings", "uris", "data", "ids", "scores"]:
                if key in res:
                    apply_mask(key)

            res["applied_distance_threshold"] = distance_threshold

        if where is not None:
            res["applied_where"] = where

        return res
