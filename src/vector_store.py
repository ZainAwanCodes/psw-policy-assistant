"""
FAISS Vector Store Module for PSW Policy Assistant.
Manages index creation, metadata mapping, safe JSON persistence, freshness checking,
and cosine similarity search on normalized embeddings.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np

from config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_FILE,
    INDEX_DIR,
    MANIFEST_FILE,
    METADATA_FILE,
)
from src.chunking import Chunk, chunk_all_documents
from src.embeddings import EmbeddingEngine
from src.ingestion import Document, ingest_all_policies

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector index with separate JSON metadata mapping and integrity validation."""

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or EmbeddingEngine.get_instance()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.manifest: Dict[str, any] = {}

    @property
    def is_ready(self) -> bool:
        """True if the index is loaded in memory and matches metadata."""
        return (
            self.index is not None
            and self.index.ntotal > 0
            and len(self.chunks) == self.index.ntotal
        )

    def is_fresh(self, docs: Dict[str, Document]) -> bool:
        """
        Checks if the persisted index on disk is valid and matches the current
        source PDFs, hashes, and embedding model.
        """
        if not (FAISS_INDEX_FILE.is_file() and METADATA_FILE.is_file() and MANIFEST_FILE.is_file()):
            return False

        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Check model name and dimension
            if manifest.get("embedding_model") != self.embedding_engine.model_name:
                logger.info("Persisted index used different embedding model. Rebuild required.")
                return False

            if manifest.get("embedding_dimension") != self.embedding_engine.dimension:
                return False

            # Check source PDF hashes for indexable documents
            persisted_hashes = manifest.get("document_hashes", {})
            for doc_id, doc in docs.items():
                if doc.is_indexable:
                    if doc_id not in persisted_hashes:
                        logger.info(f"New indexable document '{doc_id}' not in index. Rebuild required.")
                        return False
                    if persisted_hashes[doc_id] != doc.file_hash:
                        logger.info(f"Document '{doc_id}' content changed. Rebuild required.")
                        return False

            return True
        except Exception as e:
            logger.warning(f"Error checking index freshness: {e}. Rebuild recommended.")
            return False

    def build_from_chunks(self, chunks: List[Chunk], docs: Dict[str, Document]) -> bool:
        """Embeds chunks and creates a new FAISS Flat Inner Product index."""
        if not chunks:
            logger.warning("No chunks provided to build vector index.")
            return False

        logger.info(f"Building FAISS index for {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed_texts(texts, normalize=True, show_progress_bar=False)

        # Create FAISS IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_engine.dimension)
        index.add(embeddings)

        self.index = index
        self.chunks = chunks

        # Prepare manifest
        doc_hashes = {
            doc_id: doc.file_hash for doc_id, doc in docs.items() if doc.is_indexable
        }
        self.manifest = {
            "embedding_model": self.embedding_engine.model_name,
            "embedding_dimension": self.embedding_engine.dimension,
            "total_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "document_hashes": doc_hashes,
            "indexed_documents": [doc.title for doc in docs.values() if doc.is_indexable],
        }

        # Persist automatically
        self.save()
        logger.info(f"FAISS index built and saved successfully with {self.index.ntotal} vectors.")
        return True

    def save(self) -> None:
        """Persists the FAISS index, JSON metadata, and manifest safely."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        if self.index is None or not self.chunks:
            logger.warning("Cannot save empty vector store.")
            return

        # 1. Write FAISS index binary
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))

        # 2. Write metadata JSON safely (human-readable, no pickle)
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # 3. Write manifest JSON
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

        logger.info("Vector store persisted successfully.")

    def load(self) -> bool:
        """Loads index and metadata from disk, validating integrity."""
        if not (FAISS_INDEX_FILE.is_file() and METADATA_FILE.is_file()):
            logger.info("No persisted vector store found on disk.")
            return False

        try:
            # 1. Load FAISS index
            index = faiss.read_index(str(FAISS_INDEX_FILE))

            # 2. Load chunk metadata
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            chunks = [Chunk.from_dict(d) for d in chunks_data]

            # 3. Validate dimension and count match
            if index.ntotal != len(chunks):
                logger.error(
                    f"Index size mismatch: FAISS has {index.ntotal} vectors, but metadata has {len(chunks)} chunks."
                )
                return False

            if index.d != self.embedding_engine.dimension:
                logger.error(
                    f"Dimension mismatch: Index has {index.d}, model requires {self.embedding_engine.dimension}."
                )
                return False

            # 4. Load manifest if present
            if MANIFEST_FILE.is_file():
                with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)

            self.index = index
            self.chunks = chunks
            logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors.")
            return True

        except Exception as e:
            logger.error(f"Failed to load persisted vector store: {e}")
            return False

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Embeds query and searches index.
        Returns list of (Chunk, cosine_similarity_score).
        """
        if not self.is_ready:
            logger.warning("Vector store is not ready for search.")
            return []

        # Embed query with normalization
        query_vec = self.embedding_engine.embed_query(query, normalize=True)

        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []

        # FAISS search
        scores, indices = self.index.search(query_vec, k)

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append((chunk, float(score)))

        return results

    def get_diagnostics(self) -> Dict[str, any]:
        """Returns diagnostic info for UI inspection."""
        return {
            "is_ready": self.is_ready,
            "vector_count": self.index.ntotal if self.index else 0,
            "dimension": self.index.d if self.index else self.embedding_engine.dimension,
            "embedding_model": self.embedding_engine.model_name,
            "persisted_index_exists": FAISS_INDEX_FILE.is_file(),
            "manifest": self.manifest,
        }


def initialize_or_load_vector_store(
    docs: Optional[Dict[str, Document]] = None,
    force_rebuild: bool = False,
    embedding_engine: Optional[EmbeddingEngine] = None,
) -> Tuple[VectorStore, Dict[str, Document]]:
    """
    High-level helper:
    1. Ingests policies if docs not passed.
    2. Checks if valid persisted index exists and is fresh.
    3. If fresh and not force_rebuild, loads from disk.
    4. Otherwise chunks and builds index, saving to disk.
    """
    if docs is None:
        docs = ingest_all_policies()

    store = VectorStore(embedding_engine=embedding_engine)

    if not force_rebuild and store.is_fresh(docs):
        if store.load():
            return store, docs

    # Build fresh
    chunks = chunk_all_documents(docs)
    store.build_from_chunks(chunks, docs)
    return store, docs
