"""ChromaDB vector store backend."""

from __future__ import annotations

import os

from rag_eval.backends.base import Document, SearchResult, VectorStoreBackend


class ChromaDBBackend(VectorStoreBackend):
    """
    ChromaDB backend — supports in-process (ephemeral/persistent) and HTTP client modes.

    Install extras: pip install 'rag-eval-harness[chromadb]'
    """

    def __init__(
        self,
        collection_name: str = "rag_eval",
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._persist_dir = persist_directory
        self._host = host or os.getenv("CHROMA_HOST")
        self._port = port or int(os.getenv("CHROMA_PORT", "8000"))
        self._embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._client = None
        self._collection = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().encode(texts, convert_to_numpy=True).tolist()

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            if self._host:
                client = chromadb.HttpClient(host=self._host, port=self._port)
            elif self._persist_dir:
                client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                client = chromadb.EphemeralClient()

            self._client = client
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(self, documents: list[Document]) -> None:
        coll = self._get_collection()
        texts = [d.text for d in documents]
        vecs = self._embed(texts)
        coll.add(
            ids=[d.id for d in documents],
            embeddings=vecs,
            documents=texts,
            metadatas=[d.metadata for d in documents],
        )

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        coll = self._get_collection()
        vec = self._embed([query])
        results = coll.query(query_embeddings=vec, n_results=k, include=["documents", "distances", "metadatas"])
        output = []
        for rank, (doc_id, text, dist, meta) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ),
            start=1,
        ):
            output.append(
                SearchResult(
                    document=Document(id=doc_id, text=text, metadata=meta or {}),
                    score=1.0 - dist,
                    rank=rank,
                )
            )
        return output

    def delete(self, doc_ids: list[str]) -> None:
        self._get_collection().delete(ids=doc_ids)

    def clear(self) -> None:
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = None
