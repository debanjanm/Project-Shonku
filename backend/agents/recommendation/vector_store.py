"""ChromaDB vector store wrapper. Ported ~verbatim from
AIB-RecommendationAgent's services/vector_store.py.

Stores product embeddings and metadata. Uses cosine distance.
"""

from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings

from backend.agents.recommendation.models import Product

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PERSIST_DIR = str(ROOT / "data" / "products" / "chroma_db")
COLLECTION_NAME = "products"


class VectorStore:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_product(self, product: Product, embedding: np.ndarray) -> None:
        """Add or update a product in the collection (upsert)."""
        self.collection.upsert(
            ids=[product.id],
            embeddings=[embedding.tolist()],
            documents=[product.text_blob],
            metadatas=[product.to_chroma_metadata()],
        )

    def query(
        self,
        embedding: np.ndarray,
        n_results: int = 20,
        category_filter: str | None = None,
    ) -> list[dict]:
        """
        Query by embedding vector. Returns list of dicts with:
        {id, similarity_score, metadata, document}
        """
        where = {"category": category_filter} if category_filter else None
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(n_results, self.count()),
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = 1.0 - (distance / 2.0)  # cosine: distance in [0,2]
            hits.append(
                {
                    "id": doc_id,
                    "similarity_score": round(similarity, 4),
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                }
            )
        return hits

    def get_product(self, product_id: str) -> Product | None:
        """Point lookup by product ID."""
        result = self.collection.get(ids=[product_id], include=["metadatas"])
        if not result["ids"]:
            return None
        return Product.from_chroma_metadata(result["metadatas"][0])

    def get_embedding(self, product_id: str) -> np.ndarray | None:
        """Retrieve stored embedding for a product."""
        result = self.collection.get(ids=[product_id], include=["embeddings"])
        if not result["ids"]:
            return None
        return np.array(result["embeddings"][0])

    def delete_product(self, product_id: str) -> None:
        self.collection.delete(ids=[product_id])

    def list_products(self, limit: int = 100, offset: int = 0) -> list[Product]:
        """List products with optional pagination."""
        result = self.collection.get(
            include=["metadatas"],
            limit=limit,
            offset=offset,
        )
        return [Product.from_chroma_metadata(m) for m in result["metadatas"]]

    def count(self) -> int:
        return self.collection.count()

    def product_exists(self, product_id: str) -> bool:
        result = self.collection.get(ids=[product_id])
        return len(result["ids"]) > 0
