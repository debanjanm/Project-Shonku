"""Ported from AIB-RecommendationAgent's models/product.py + models/search.py."""

from pydantic import BaseModel


class Product(BaseModel):
    id: str
    name: str
    description: str
    category: str
    price: float
    brand: str
    tags: list[str]
    image_filename: str

    @property
    def text_blob(self) -> str:
        """Full text representation for CLIP embedding."""
        return f"{self.name}. {self.brand}. {self.description}. Category: {self.category}. Tags: {', '.join(self.tags)}"

    def to_chroma_metadata(self) -> dict:
        """Flatten to ChromaDB-compatible metadata (scalar values only)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "price": self.price,
            "brand": self.brand,
            "tags": ",".join(self.tags),
            "image_filename": self.image_filename,
        }

    @classmethod
    def from_chroma_metadata(cls, metadata: dict) -> "Product":
        """Reconstruct from ChromaDB flat metadata."""
        return cls(
            id=metadata["id"],
            name=metadata["name"],
            description=metadata["description"],
            category=metadata["category"],
            price=float(metadata["price"]),
            brand=metadata["brand"],
            tags=metadata["tags"].split(",") if metadata["tags"] else [],
            image_filename=metadata["image_filename"],
        )


class QueryExpansion(BaseModel):
    expanded_query: str
    extracted_intent: dict
    search_filters: dict
