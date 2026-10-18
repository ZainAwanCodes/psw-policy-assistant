
"""
PDF Ingestion Module for PSW Policy Assistant.
Handles locating policy PDFs, page-by-page text extraction, normalization,
metadata recording, and detection of scanned or unreadable documents.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DATA_DIR, POLICIES_DIR, POLICY_DEFINITIONS

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class PageExtraction:
    """Represents text extracted from a single page with preserved page index."""
    page_number: int  # 1-indexed
    text: str
    raw_char_count: int
    is_empty: bool


@dataclass
class Document:
    """Represents an ingested policy document with complete metadata."""
    document_id: str
    title: str
    source_filename: str
    file_path: Optional[Path]
    file_hash: str
    file_mtime: float
    total_pages: int
    pages: List[PageExtraction] = field(default_factory=list)
    total_chars: int = 0
    status: str = "pending"  # "indexed", "scanned_ocr_required", "missing", "unreadable"
    status_message: str = ""
    is_scanned: bool = False

    @property
    def is_indexable(self) -> bool:
        return self.status == "indexed" and self.total_chars > 0


def compute_file_sha256(path: Path) -> str:
    """Calculates SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_page_text(raw_text: str) -> str:
    """
    Normalizes whitespace and removes extraction artifacts while preserving
    paragraph breaks, clause enumerations, and punctuation.
    """
    if not raw_text:
        return ""

    # Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace non-breaking spaces and special zero-width spaces
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Replace unicode quotes and apostrophes cleanly
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Normalize horizontal whitespace (tabs, repeated spaces)
    text = re.sub(r"[ \t]+", " ", text)

    # Clean leading/trailing spaces per line
    lines = [line.strip() for line in text.split("\n")]

    # Collapse excessive blank lines to preserve clean paragraph breaks
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)

    normalized = "\n".join(cleaned_lines).strip()
    return normalized


def locate_policy_file(doc_id: str, doc_def: Dict[str, any]) -> Optional[Path]:
    """
    Locates the policy PDF file either in POLICIES_DIR (data/policies)
    or directly in DATA_DIR (data/) matching primary or alternate names.
    """
    search_dirs = [POLICIES_DIR, DATA_DIR]
    candidates = [doc_def["primary_filename"]] + doc_def.get("alternate_filenames", [])

    for directory in search_dirs:
        if not directory.exists():
            continue
        for candidate in candidates:
            target = directory / candidate
            if target.is_file():
                return target

    # Case-insensitive search fallback
    for directory in search_dirs:
        if not directory.exists():
            continue
        try:
            for entry in directory.iterdir():
                if entry.is_file() and entry.suffix.lower() == ".pdf":
                    for candidate in candidates:
                        if entry.name.lower() == candidate.lower():
                            return entry
        except Exception:
            pass

    return None


def extract_pages_pymupdf(file_path: Path) -> Tuple[List[PageExtraction], int]:
    """Extracts text page by page using PyMuPDF."""
    import pymupdf

    doc = pymupdf.open(file_path)
    pages: List[PageExtraction] = []
    total_chars = 0

    for idx, page in enumerate(doc):
        raw_text = page.get_text() or ""
        normalized = normalize_page_text(raw_text)
        is_empty = len(normalized.strip()) == 0
        pages.append(
            PageExtraction(
                page_number=idx + 1,
                text=normalized,
                raw_char_count=len(raw_text),
                is_empty=is_empty,
            )
        )
        total_chars += len(normalized)

    total_pages = len(doc)
    doc.close()
    return pages, total_chars


def extract_pages_pypdf(file_path: Path) -> Tuple[List[PageExtraction], int]:
    """Fallback text extractor using pypdf."""
    import pypdf

    reader = pypdf.PdfReader(str(file_path))
    pages: List[PageExtraction] = []
    total_chars = 0

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        normalized = normalize_page_text(raw_text)
        is_empty = len(normalized.strip()) == 0
        pages.append(
            PageExtraction(
                page_number=idx + 1,
                text=normalized,
                raw_char_count=len(raw_text),
                is_empty=is_empty,
            )
        )
        total_chars += len(normalized)

    return pages, total_chars


def ingest_single_document(doc_id: str, doc_def: Dict[str, any]) -> Document:
    """
    Ingests a single policy document, extracting text with page preservation
    and detecting scanned or empty files.
    """
    title = doc_def["title"]
    file_path = locate_policy_file(doc_id, doc_def)

    if not file_path:
        logger.warning(f"Policy file not found for '{title}' (ID: {doc_id})")
        return Document(
            document_id=doc_id,
            title=title,
            source_filename=doc_def["primary_filename"],
            file_path=None,
            file_hash="",
            file_mtime=0.0,
            total_pages=0,
            status="missing",
            status_message=f"File '{doc_def['primary_filename']}' not found in data/ or data/policies/",
        )

    try:
        file_hash = compute_file_sha256(file_path)
        file_mtime = file_path.stat().st_mtime
    except Exception as e:
        logger.error(f"Error reading file stats for {file_path}: {e}")
        return Document(
            document_id=doc_id,
            title=title,
            source_filename=file_path.name,
            file_path=file_path,
            file_hash="",
            file_mtime=0.0,
            total_pages=0,
            status="unreadable",
            status_message=f"Cannot access file: {e}",
        )

    # Attempt extraction via PyMuPDF first, fallback to pypdf
    pages: List[PageExtraction] = []
    total_chars = 0
    extraction_error = None

    try:
        pages, total_chars = extract_pages_pymupdf(file_path)
    except Exception as e_pymupdf:
        logger.warning(f"PyMuPDF extraction failed for {file_path.name}: {e_pymupdf}. Trying pypdf...")
        try:
            pages, total_chars = extract_pages_pypdf(file_path)
        except Exception as e_pypdf:
            extraction_error = f"Extraction failed (PyMuPDF: {e_pymupdf}; pypdf: {e_pypdf})"
            logger.error(extraction_error)

    if extraction_error:
        return Document(
            document_id=doc_id,
            title=title,
            source_filename=file_path.name,
            file_path=file_path,
            file_hash=file_hash,
            file_mtime=file_mtime,
            total_pages=len(pages),
            status="unreadable",
            status_message=extraction_error,
        )

    # Check if document has 0 extractable characters (scanned document)
    if total_chars == 0:
        # Check if an OCR sidecar exists in data/ocr_sidecars/{doc_id}.json
        sidecar_path = DATA_DIR / "ocr_sidecars" / f"{doc_id}.json"
        if sidecar_path.is_file():
            try:
                import json
                with open(sidecar_path, "r", encoding="utf-8") as sf:
                    sidecar_data = json.load(sf)
                sidecar_pages = []
                sidecar_chars = 0
                for item in sidecar_data:
                    p_num = item.get("page_number", len(sidecar_pages) + 1)
                    p_txt = normalize_page_text(item.get("text", ""))
                    sidecar_chars += len(p_txt)
                    sidecar_pages.append(
                        PageExtraction(
                            page_number=p_num,
                            text=p_txt,
                            raw_char_count=len(item.get("text", "")),
                            is_empty=len(p_txt.strip()) == 0,
                        )
                    )
                if sidecar_chars > 0:
                    logger.info(
                        f"Loaded OCR sidecar for '{title}' ({sidecar_path.name}): {len(sidecar_pages)} pages, {sidecar_chars:,} chars."
                    )
                    return Document(
                        document_id=doc_id,
                        title=title,
                        source_filename=file_path.name,
                        file_path=file_path,
                        file_hash=file_hash,
                        file_mtime=file_mtime,
                        total_pages=len(sidecar_pages),
                        pages=sidecar_pages,
                        total_chars=sidecar_chars,
                        status="indexed",
                        status_message=f"Indexed via OCR sidecar: {len(sidecar_pages)} pages, {sidecar_chars:,} characters.",
                        is_scanned=True,
                    )
            except Exception as e_sidecar:
                logger.warning(f"Error loading sidecar for {doc_id}: {e_sidecar}")

        logger.warning(
            f"Document '{title}' ({file_path.name}) contains 0 extractable text characters across {len(pages)} pages. "
            "It appears to be a scanned image-only PDF. OCR is required to extract text."
        )
        return Document(
            document_id=doc_id,
            title=title,
            source_filename=file_path.name,
            file_path=file_path,
            file_hash=file_hash,
            file_mtime=file_mtime,
            total_pages=len(pages),
            pages=pages,
            total_chars=0,
            status="scanned_ocr_required",
            status_message=(
                f"Scanned image PDF ({len(pages)} pages). Contains no digital text layer. "
                "Optical Character Recognition (OCR) is required to index this document."
            ),
            is_scanned=True,
        )

    logger.info(
        f"Successfully extracted '{title}': {len(pages)} pages, {total_chars:,} characters."
    )
    return Document(
        document_id=doc_id,
        title=title,
        source_filename=file_path.name,
        file_path=file_path,
        file_hash=file_hash,
        file_mtime=file_mtime,
        total_pages=len(pages),
        pages=pages,
        total_chars=total_chars,
        status="indexed",
        status_message=f"Ready: {len(pages)} pages, {total_chars:,} characters extracted.",
        is_scanned=False,
    )


def ingest_all_policies() -> Dict[str, Document]:
    """
    Ingests all four defined PSW policies.
    Returns dictionary mapping document_id to Document.
    """
    results: Dict[str, Document] = {}
    for doc_id, doc_def in POLICY_DEFINITIONS.items():
        results[doc_id] = ingest_single_document(doc_id, doc_def)
    return results
