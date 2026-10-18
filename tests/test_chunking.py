"""
Unit tests for text chunking module.
"""

import pytest
from src.chunking import (
    Chunk,
    chunk_all_documents,
    chunk_document,
    chunk_page,
    split_paragraph_into_sentences,
    split_text_into_paragraphs,
)
from src.ingestion import Document, PageExtraction


def test_split_text_into_paragraphs():
    text = "Paragraph 1 line 1.\nParagraph 1 line 2.\n\nParagraph 2 is here."
    paras = split_text_into_paragraphs(text)
    assert len(paras) == 2
    assert paras[0] == "Paragraph 1 line 1.\nParagraph 1 line 2."
    assert paras[1] == "Paragraph 2 is here."


def test_chunk_page_metadata_preservation():
    page = PageExtraction(
        page_number=3,
        text="Section 3.1 Policy detail.\n\nSection 3.2 Another detail for PSW employees.",
        raw_char_count=80,
        is_empty=False,
    )
    chunks = chunk_page(
        page=page,
        doc_id="test_doc",
        doc_title="Test Policy",
        source_filename="test_policy.pdf",
        target_chars=200,
        overlap_chars=30,
    )

    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "test_doc"
        assert c.document_title == "Test Policy"
        assert c.source_filename == "test_policy.pdf"
        assert c.page_number == 3
        assert "test_doc_p3_c" in c.chunk_id
        assert c.char_count > 0
        assert c.token_count_approx > 0


def test_chunk_document_skips_unindexable():
    doc = Document(
        document_id="scanned_doc",
        title="Scanned Policy",
        source_filename="scanned.pdf",
        file_path=None,
        file_hash="",
        file_mtime=0.0,
        total_pages=5,
        total_chars=0,
        status="scanned_ocr_required",
        is_scanned=True,
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 0
