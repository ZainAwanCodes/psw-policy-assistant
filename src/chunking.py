"""
Chunking Module for PSW Policy Assistant.
Implements page-aware, paragraph-preserving text splitting with strict metadata retention.
"""

import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from config import (
    APPROX_CHARS_PER_TOKEN,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
)
from src.ingestion import Document, PageExtraction


@dataclass
class Chunk:
    """Represents a text chunk strictly tied to its source document and page number."""
    chunk_id: str
    document_id: str
    document_title: str
    source_filename: str
    page_number: int
    text: str
    token_count_approx: int
    char_count: int
    chunk_index: int

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "Chunk":
        return cls(**data)


def split_text_into_paragraphs(text: str) -> List[str]:
    """Splits page text into coherent paragraphs and section blocks."""
    if not text:
        return []

    # Split by double newline or clause numbers at start of lines (e.g. "3.4. CONFLICT")
    raw_paras = re.split(r"\n\s*\n+", text)
    paras = [p.strip() for p in raw_paras if p.strip()]
    return paras


def split_paragraph_into_sentences(para: str) -> List[str]:
    """Splits a large paragraph into sentences."""
    # Split on sentence terminals followed by space
    sentences = re.split(r"(?<=[.!?])\s+", para)
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(
    page: PageExtraction,
    doc_id: str,
    doc_title: str,
    source_filename: str,
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> List[Chunk]:
    """
    Splits text from a single page into overlapping chunks while preserving
    page boundaries and section structures.
    """
    if not page.text or page.is_empty:
        return []

    paragraphs = split_text_into_paragraphs(page.text)
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    current_pieces: List[str] = []
    current_len = 0
    chunk_index = 0

    for para in paragraphs:
        # If single paragraph exceeds 1.5x target, break it down by sentences
        if len(para) > (target_chars * 1.5):
            sentences = split_paragraph_into_sentences(para)
            for sentence in sentences:
                if current_len + len(sentence) > target_chars and current_pieces:
                    # Flush current chunk
                    chunk_text = "\n\n".join(current_pieces).strip()
                    if chunk_text:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{doc_id}_p{page.page_number}_c{chunk_index}",
                                document_id=doc_id,
                                document_title=doc_title,
                                source_filename=source_filename,
                                page_number=page.page_number,
                                text=chunk_text,
                                token_count_approx=max(1, len(chunk_text) // APPROX_CHARS_PER_TOKEN),
                                char_count=len(chunk_text),
                                chunk_index=chunk_index,
                            )
                        )
                        chunk_index += 1

                    # Keep overlap from the end of current_pieces
                    overlap_piece = sentence[-overlap_chars:] if len(sentence) > overlap_chars else sentence
                    current_pieces = [overlap_piece]
                    current_len = len(overlap_piece)
                else:
                    current_pieces.append(sentence)
                    current_len += len(sentence) + 2
        else:
            if current_len + len(para) > target_chars and current_pieces:
                chunk_text = "\n\n".join(current_pieces).strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}_p{page.page_number}_c{chunk_index}",
                            document_id=doc_id,
                            document_title=doc_title,
                            source_filename=source_filename,
                            page_number=page.page_number,
                            text=chunk_text,
                            token_count_approx=max(1, len(chunk_text) // APPROX_CHARS_PER_TOKEN),
                            char_count=len(chunk_text),
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

                # Overlap: keep the last paragraph if small enough, or a slice of it
                last_piece = current_pieces[-1]
                if len(last_piece) > overlap_chars:
                    last_piece = last_piece[-overlap_chars:]
                current_pieces = [last_piece, para]
                current_len = len(last_piece) + len(para) + 2
            else:
                current_pieces.append(para)
                current_len += len(para) + 2

    # Flush remaining text
    if current_pieces:
        chunk_text = "\n\n".join(current_pieces).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_p{page.page_number}_c{chunk_index}",
                    document_id=doc_id,
                    document_title=doc_title,
                    source_filename=source_filename,
                    page_number=page.page_number,
                    text=chunk_text,
                    token_count_approx=max(1, len(chunk_text) // APPROX_CHARS_PER_TOKEN),
                    char_count=len(chunk_text),
                    chunk_index=chunk_index,
                )
            )

    return chunks


def chunk_document(
    doc: Document,
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> List[Chunk]:
    """Chunks all pages of an ingested document."""
    if not doc.is_indexable:
        return []

    doc_chunks: List[Chunk] = []
    for page in doc.pages:
        page_chunks = chunk_page(
            page=page,
            doc_id=doc.document_id,
            doc_title=doc.title,
            source_filename=doc.source_filename,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        doc_chunks.extend(page_chunks)

    return doc_chunks


def chunk_all_documents(
    docs: Dict[str, Document],
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> List[Chunk]:
    """Processes all indexable documents and produces a flat list of verified chunks."""
    all_chunks: List[Chunk] = []
    for doc in docs.values():
        if doc.is_indexable:
            all_chunks.extend(chunk_document(doc, target_chars, overlap_chars))
    return all_chunks
