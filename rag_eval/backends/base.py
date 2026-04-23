"""Abstract base class for all vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


class VectorStoreBackend(ABC):
    """
    Plug-and-play interface for vector store backends.

    Implementations: FAISSBackend, QdrantBackend, ChromaDBBackend, WeaviateBackend.
    """

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Embed and index a list of documents."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Return the top-K most similar documents for a query string."""

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> None:
        """Remove documents by ID from the index."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the index."""

    def search_ids(self, query: str, k: int = 10) -> list[str]:
        """Convenience: return only the doc IDs from a search."""
        return [r.document.id for r in self.search(query, k)]
