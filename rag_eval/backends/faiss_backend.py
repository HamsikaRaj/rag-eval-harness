"""FAISS in-memory vector store backend."""

from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np

from rag_eval.backends.base import Document, SearchResult, VectorStoreBackend


class FAISSBackend(VectorStoreBackend):
    """
    Local FAISS flat-L2 index.  No server required — great for CI and quick experiments.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        index_path: str | None = None,
    ) -> None:
        self._embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._index_path = index_path
        self._embedder = None
        self._index = None
        self._id_map: list[str] = []
        self._docs: dict[str, Document] = {}

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _get_index(self, dim: int):
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatL2(dim)
        return self._index

    def _embed(self, texts: list[str]) -> np.ndarray:
        embedder = self._get_embedder()
        vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype("float32")

    def add_documents(self, documents: list[Document]) -> None:
        texts = [d.text for d in documents]
        vecs = self._embed(texts)
        index = self._get_index(vecs.shape[1])
        index.add(vecs)
        for doc in documents:
            if not doc.id:
                doc.id = str(uuid.uuid4())
            self._id_map.append(doc.id)
            self._docs[doc.id] = doc

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        vec = self._embed([query])
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx == -1:
                continue
            doc_id = self._id_map[idx]
            results.append(
                SearchResult(
                    document=self._docs[doc_id],
                    score=float(1.0 / (1.0 + dist)),
                    rank=rank,
                )
            )
        return results

    def delete(self, doc_ids: list[str]) -> None:
        # FAISS flat index doesn't support selective deletion; rebuild without those IDs
        keep_ids = [d for d in self._id_map if d not in set(doc_ids)]
        keep_docs = [self._docs[d] for d in keep_ids]
        self.clear()
        if keep_docs:
            self.add_documents(keep_docs)

    def clear(self) -> None:
        self._index = None
        self._id_map = []
        self._docs = {}

    def save(self, path: str | None = None) -> None:
        import faiss, pickle

        out = path or self._index_path
        if not out:
            raise ValueError("Provide a path to save the FAISS index")
        faiss.write_index(self._index, out + ".index")
        with open(out + ".meta", "wb") as f:
            pickle.dump({"id_map": self._id_map, "docs": self._docs}, f)

    def load(self, path: str | None = None) -> None:
        import faiss, pickle

        src = path or self._index_path
        if not src:
            raise ValueError("Provide a path to load the FAISS index")
        self._index = faiss.read_index(src + ".index")
        with open(src + ".meta", "rb") as f:
            meta = pickle.load(f)
        self._id_map = meta["id_map"]
        self._docs = meta["docs"]
