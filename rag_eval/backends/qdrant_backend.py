"""Qdrant vector store backend."""

from __future__ import annotations

import os
import uuid

from rag_eval.backends.base import Document, SearchResult, VectorStoreBackend


class QdrantBackend(VectorStoreBackend):
    """
    Qdrant backend — supports both local (in-memory) and remote server modes.

    Install extras: pip install 'rag-eval-harness[qdrant]'
    """

    def __init__(
        self,
        collection_name: str = "rag_eval",
        url: str | None = None,
        api_key: str | None = None,
        embedding_model: str | None = None,
        in_memory: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self._url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._api_key = api_key or os.getenv("QDRANT_API_KEY")
        self._embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._in_memory = in_memory
        self._client = None
        self._embedder = None
        self._vector_size: int | None = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().encode(texts, convert_to_numpy=True).tolist()

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            if self._in_memory:
                self._client = QdrantClient(":memory:")
            else:
                self._client = QdrantClient(url=self._url, api_key=self._api_key or None)
        return self._client

    def _ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        existing = [c.name for c in client.get_collections().collections]
        if self.collection_name not in existing:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_documents(self, documents: list[Document]) -> None:
        from qdrant_client.models import PointStruct

        texts = [d.text for d in documents]
        vecs = self._embed(texts)
        self._vector_size = len(vecs[0])
        self._ensure_collection(self._vector_size)

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.id or str(i))),
                vector=vecs[i],
                payload={"doc_id": doc.id, "text": doc.text, **doc.metadata},
            )
            for i, doc in enumerate(documents)
        ]
        self._get_client().upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        vec = self._embed([query])[0]
        hits = self._get_client().search(
            collection_name=self.collection_name, query_vector=vec, limit=k
        )
        return [
            SearchResult(
                document=Document(
                    id=hit.payload.get("doc_id", str(hit.id)),
                    text=hit.payload.get("text", ""),
                    metadata={k: v for k, v in hit.payload.items() if k not in ("doc_id", "text")},
                ),
                score=hit.score,
                rank=rank,
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    def delete(self, doc_ids: list[str]) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        self._get_client().delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
            ),
        )

    def clear(self) -> None:
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._vector_size = None
