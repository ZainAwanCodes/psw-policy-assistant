"""
Embeddings Module for PSW Policy Assistant.
Provides local, normalized vector embeddings using SentenceTransformers.
Loads once, reuses in-memory model, and computes cosine-ready L2-normalized vectors.
"""

import logging
from typing import List, Optional, Union
import numpy as np

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Wrapper around SentenceTransformer with caching, normalization, and safety guards."""

    _instance: Optional["EmbeddingEngine"] = None
    _model = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self._load_model()

    def _load_model(self):
        """Loads the sentence-transformer model if not already in memory."""
        if EmbeddingEngine._model is None:
            logger.info(f"Loading embedding model: '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            EmbeddingEngine._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully.")
        self.model = EmbeddingEngine._model

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "EmbeddingEngine":
        """Singleton accessor to avoid reloading the model multiple times."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Computes dense embeddings for a list of strings.
        Returns a 2D float32 numpy array of shape (N, dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Sanitize empty or non-string inputs
        sanitized_texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]

        embeddings = self.model.encode(
            sanitized_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Computes embedding for a single search query.
        Returns a 2D float32 numpy array of shape (1, dimension).
        """
        query_str = query.strip() if isinstance(query, str) else ""
        if not query_str:
            # Return zero vector if query is empty
            vec = np.zeros((1, self.dimension), dtype=np.float32)
            return vec

        embedding = self.model.encode(
            [query_str],
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)
