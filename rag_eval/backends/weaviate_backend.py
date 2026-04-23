"""Weaviate vector store backend."""

from __future__ import annotations

import os

from rag_eval.backends.base import Document, SearchResult, VectorStoreBackend


class WeaviateBackend(VectorStoreBackend):
    """
    Weaviate v4 client backend.

    Install extras: pip install 'rag-eval-harness[weaviate]'
    """

    def __init__(
        self,
        class_name: str = "RagEvalDoc",
        url: str | None = None,
        api_key: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._class_name = class_name
        self._url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self._api_key = api_key or os.getenv("WEAVIATE_API_KEY")
        self._embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._client = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().encode(texts, convert_to_numpy=True).tolist()

    def _get_client(self):
        if self._client is None:
            import weaviate
            from weaviate.auth import AuthApiKey

            auth = AuthApiKey(self._api_key) if self._api_key else None
            self._client = weaviate.connect_to_custom(
                http_host=self._url.split("://")[-1].split(":")[0],
                http_port=int(self._url.split(":")[-1]) if ":" in self._url.split("://")[-1] else 8080,
                http_secure=self._url.startswith("https"),
                grpc_host=self._url.split("://")[-1].split(":")[0],
                grpc_port=50051,
                grpc_secure=False,
                auth_credentials=auth,
            )
            self._ensure_class()
        return self._client

    def _ensure_class(self) -> None:
        import weaviate.classes.config as wvc

        client = self._client
        if not client.collections.exists(self._class_name):
            client.collections.create(
                name=self._class_name,
                properties=[
                    wvc.Property(name="doc_id", data_type=wvc.DataType.TEXT),
                    wvc.Property(name="text", data_type=wvc.DataType.TEXT),
                ],
                vectorizer_config=wvc.Configure.Vectorizer.none(),
            )

    def add_documents(self, documents: list[Document]) -> None:
        client = self._get_client()
        collection = client.collections.get(self._class_name)
        texts = [d.text for d in documents]
        vecs = self._embed(texts)

        with collection.batch.dynamic() as batch:
            for doc, vec in zip(documents, vecs):
                batch.add_object(
                    properties={"doc_id": doc.id, "text": doc.text, **doc.metadata},
                    vector=vec,
                )

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        from weaviate.classes.query import MetadataQuery

        client = self._get_client()
        collection = client.collections.get(self._class_name)
        vec = self._embed([query])[0]
        results = collection.query.near_vector(
            near_vector=vec,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            SearchResult(
                document=Document(
                    id=obj.properties.get("doc_id", str(obj.uuid)),
                    text=obj.properties.get("text", ""),
                    metadata={k: v for k, v in obj.properties.items() if k not in ("doc_id", "text")},
                ),
                score=1.0 - (obj.metadata.distance or 0.0),
                rank=rank,
            )
            for rank, obj in enumerate(results.objects, start=1)
        ]

    def delete(self, doc_ids: list[str]) -> None:
        from weaviate.classes.query import Filter

        client = self._get_client()
        collection = client.collections.get(self._class_name)
        for doc_id in doc_ids:
            collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))

    def clear(self) -> None:
        client = self._get_client()
        client.collections.delete(self._class_name)
        self._ensure_class()
