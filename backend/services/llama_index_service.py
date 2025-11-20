"""LlamaIndex integration service for retrieval"""

from __future__ import annotations

from typing import List, Dict, Optional
import logging

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


class LlamaIndexService:
    """Wraps LlamaIndex vector store for retrieval."""

    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self.index: Optional[VectorStoreIndex] = None
        self._ensure_embed_model()

    def _ensure_embed_model(self) -> None:
        """Configure embedding model for LlamaIndex."""
        try:
            Settings.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
            logger.info("Initialized LlamaIndex embedding model: %s", self.embed_model_name)
        except Exception as exc:
            logger.error("Failed to initialize LlamaIndex embedding model: %s", exc)
            raise

    def build_index(self, chunks: List[Dict]) -> None:
        """Create a vector index from processed chunks."""
        if not chunks:
            raise ValueError("No chunks provided to build LlamaIndex")

        nodes: List[TextNode] = []
        for chunk in chunks:
            metadata = {
                "chunk_id": chunk.get("id"),
                "document_name": chunk.get("document_name"),
                "page_number": chunk.get("page_number"),
                "citation": chunk.get("citation"),
            }

            node = TextNode(
                text=chunk["text"],
                id_=str(chunk.get("id")),
                metadata=metadata,
                embedding=chunk.get("embedding"),
            )
            nodes.append(node)

        self.index = VectorStoreIndex(nodes, show_progress=False)
        logger.info("Built LlamaIndex with %d nodes", len(nodes))

    def is_ready(self) -> bool:
        """Return True if the index is initialized."""
        return self.index is not None

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant chunks using LlamaIndex."""
        if self.index is None:
            raise ValueError("LlamaIndex is not initialized")

        retriever = self.index.as_retriever(similarity_top_k=max(k, 1))
        results = retriever.retrieve(query)

        formatted_chunks: List[Dict] = []
        for rank, node_with_score in enumerate(results, start=1):
            node, score = self._extract_node_and_score(node_with_score)
            metadata = node.metadata or {}

            formatted_chunks.append(
                {
                    "id": metadata.get("chunk_id"),
                    "text": node.get_content(metadata_mode="all"),
                    "document_name": metadata.get("document_name", "Unknown"),
                    "page_number": metadata.get("page_number", 0),
                    "citation": metadata.get("citation", ""),
                    "final_score": round(score or 0.0, 4),
                    "rank": rank,
                    "retrieval_backend": "llama_index",
                    "token_count": None,
                }
            )

        return formatted_chunks

    @staticmethod
    def _extract_node_and_score(node_with_score: NodeWithScore):
        """Normalize NodeWithScore outputs across LlamaIndex versions."""
        if hasattr(node_with_score, "node"):
            node = node_with_score.node
            score = getattr(node_with_score, "score", 0.0)
        else:
            node = node_with_score
            score = getattr(node_with_score, "score", 0.0)
        return node, score


