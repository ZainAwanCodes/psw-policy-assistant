"""
Retrieval Module for PSW Policy Assistant.
Encapsulates semantic query search, ranking, evidence selection, and diagnostic metadata.
Completely decoupled from LLM answer generation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from config import DEFAULT_TOP_K, MIN_SIMILARITY_SCORE
from src.chunking import Chunk
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    """Represents a retrieved chunk with similarity score and citation formatting."""
    chunk: Chunk
    score: float
    rank: int

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def document_title(self) -> str:
        return self.chunk.document_title

    @property
    def page_number(self) -> int:
        return self.chunk.page_number

    @property
    def source_filename(self) -> str:
        return self.chunk.source_filename

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def citation_label(self) -> str:
        return f"{self.document_title} — Page {self.page_number}"


@dataclass
class RetrievalResult:
    """Encapsulates the complete retrieval response and diagnostic metrics."""
    query: str
    passages: List[RetrievedPassage]
    top_k_requested: int
    total_returned: int

    @property
    def is_empty(self) -> bool:
        return len(self.passages) == 0

    @property
    def max_score(self) -> float:
        return max((p.score for p in self.passages), default=0.0)

    @property
    def unique_citations(self) -> List[str]:
        """Returns ordered list of unique 'Document — Page X' citations."""
        seen: Set[str] = set()
        citations: List[str] = []
        for p in self.passages:
            label = p.citation_label
            if label not in seen:
                seen.add(label)
                citations.append(label)
        return citations

    def get_diagnostic_dict(self) -> Dict[str, any]:
        """Returns structured diagnostic dictionary for dev/evaluation views."""
        return {
            "query": self.query,
            "passages_count": len(self.passages),
            "max_score": round(self.max_score, 4),
            "citations": self.unique_citations,
            "passages": [
                {
                    "rank": p.rank,
                    "score": round(p.score, 4),
                    "document": p.document_title,
                    "page": p.page_number,
                    "chunk_id": p.chunk.chunk_id,
                    "text_preview": p.text[:150] + ("..." if len(p.text) > 150 else ""),
                }
                for p in self.passages
            ],
        }


def retrieve_evidence(
    query: str,
    vector_store: VectorStore,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = MIN_SIMILARITY_SCORE,
) -> RetrievalResult:
    """
    Performs cosine-similarity search against the vector store and formats retrieved passages.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return RetrievalResult(
            query=clean_query,
            passages=[],
            top_k_requested=top_k,
            total_returned=0,
        )

    raw_results = vector_store.search(clean_query, top_k=top_k)

    passages: List[RetrievedPassage] = []
    for rank, (chunk, score) in enumerate(raw_results, start=1):
        if score >= min_score:
            passages.append(RetrievedPassage(chunk=chunk, score=score, rank=rank))

    return RetrievalResult(
        query=clean_query,
        passages=passages,
        top_k_requested=top_k,
        total_returned=len(passages),
    )


def format_evidence_block(passages: List[RetrievedPassage]) -> str:
    """
    Formats retrieved passages into a structured text block for the LLM prompt.
    Each passage explicitly labels its source document and page number.
    """
    if not passages:
        return "No relevant passages found in the indexed policy documents."

    blocks = []
    for p in passages:
        block = (
            f"--- START EVIDENCE ---\n"
            f"Document: {p.document_title}\n"
            f"Page: {p.page_number}\n"
            f"Source File: {p.source_filename}\n"
            f"Content:\n{p.text}\n"
            f"--- END EVIDENCE ---"
        )
        blocks.append(block)

    return "\n\n".join(blocks)
