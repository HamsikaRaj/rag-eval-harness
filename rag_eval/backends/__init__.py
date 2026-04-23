from rag_eval.backends.base import VectorStoreBackend, Document, SearchResult
from rag_eval.backends.faiss_backend import FAISSBackend

__all__ = ["VectorStoreBackend", "Document", "SearchResult", "FAISSBackend", "get_backend"]


def get_backend(name: str, **kwargs) -> VectorStoreBackend:
    """Factory: instantiate a backend by name ('faiss', 'qdrant', 'chromadb', 'weaviate')."""
    name = name.lower()
    if name == "faiss":
        return FAISSBackend(**kwargs)
    if name == "qdrant":
        from rag_eval.backends.qdrant_backend import QdrantBackend
        return QdrantBackend(**kwargs)
    if name in ("chromadb", "chroma"):
        from rag_eval.backends.chromadb_backend import ChromaDBBackend
        return ChromaDBBackend(**kwargs)
    if name == "weaviate":
        from rag_eval.backends.weaviate_backend import WeaviateBackend
        return WeaviateBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Choose from: faiss, qdrant, chromadb, weaviate")
