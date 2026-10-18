"""
Unit tests for PDF Ingestion module.
"""

from pathlib import Path
import pytest
from src.ingestion import (
    Document,
    PageExtraction,
    compute_file_sha256,
    ingest_all_policies,
    ingest_single_document,
    normalize_page_text,
)


def test_normalize_page_text():
    raw = "  Hello   world! \r\n\r\n\r\n\r\nParagraph \xa0 2 with   spaces.  \n"
    normalized = normalize_page_text(raw)
    assert "Hello world!" in normalized
    assert "Paragraph 2 with spaces." in normalized
    assert "\r" not in normalized
    assert "\xa0" not in normalized


def test_missing_document_handling():
    dummy_def = {
        "title": "Non-existent Policy",
        "primary_filename": "non_existent_file_12345.pdf",
    }
    doc = ingest_single_document("missing_doc", dummy_def)
    assert doc.status == "missing"
    assert doc.is_indexable is False
    assert doc.total_pages == 0
    assert "not found" in doc.status_message.lower()


def test_ingest_all_real_policies():
    docs = ingest_all_policies()
    assert len(docs) == 4
    assert "general_code_of_conduct" in docs
    assert "whistle_blowing_policy" in docs
    assert "anti_corruption_policy" in docs
    assert "harassment_policy" in docs

    # General Code of Conduct is digital and indexable
    gcc = docs["general_code_of_conduct"]
    assert gcc.status == "indexed"
    assert gcc.is_indexable is True
    assert gcc.total_pages == 5
    assert gcc.total_chars > 5000
    assert len(gcc.pages) == 5
    assert gcc.pages[0].page_number == 1

    # Scanned documents are detected without crashing
    ac = docs["anti_corruption_policy"]
    assert ac.is_scanned is True
    assert ac.status == "scanned_ocr_required"
    assert ac.is_indexable is False
    assert "ocr" in ac.status_message.lower()
    assert "required" in ac.status_message.lower()
