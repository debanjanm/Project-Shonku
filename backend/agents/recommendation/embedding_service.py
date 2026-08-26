"""CLIP-based embedding service using open_clip. Ported ~verbatim from
AIB-RecommendationAgent's services/embedding_service.py (print -> logging).

Loads the model once at startup. Provides text, image, and combined embeddings.
"""

import logging
from functools import lru_cache

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        """Encode text to a normalized 512-d CLIP vector."""
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image to a normalized 512-d CLIP vector."""
        with torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def combine(
        self,
        text_emb: np.ndarray,
        image_emb: np.ndarray,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
    ) -> np.ndarray:
        """Weighted combination of text and image embeddings, re-normalized."""
        combined = text_weight * text_emb + image_weight * image_emb
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined

    def embed_query(
        self,
        text: str,
        image: Image.Image | None = None,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
    ) -> np.ndarray:
        """
        Embed a user query. If image is provided, combines text+image.
        Uses 60/40 text/image weighting by default (text is enriched by LLM expansion).
        """
        text_emb = self.embed_text(text)
        if image is None:
            return text_emb
        image_emb = self.embed_image(image)
        return self.combine(text_emb, image_emb, text_weight, image_weight)

    def embed_product(self, text: str, image: Image.Image | None = None) -> np.ndarray:
        """
        Embed a product at index time. Combines text+image with equal weight.
        Falls back to text-only if no image is provided.
        """
        text_emb = self.embed_text(text)
        if image is None:
            return text_emb
        image_emb = self.embed_image(image)
        return self.combine(text_emb, image_emb, 0.5, 0.5)


@lru_cache(maxsize=1)
def get_embedding_service(
    model_name: str = "ViT-B-32", pretrained: str = "openai"
) -> EmbeddingService:
    """Singleton factory — loads CLIP once per process."""
    logger.info("loading CLIP model %s (%s)...", model_name, pretrained)
    return EmbeddingService(model_name=model_name, pretrained=pretrained)
