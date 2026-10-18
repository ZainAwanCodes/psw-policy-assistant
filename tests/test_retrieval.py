"""
Unit tests for FAISS vector store and retrieval module.
"""

import os
from pathlib import Path
import pytest
from src.chunking import Chunk
from src.embeddings import EmbeddingEngine
from src.retrieval import RetrievalResult, format_evidence_block, retrieve_evidence
from src.vector_store import VectorStore, initialize_or_load_vector_store


@pytest.fixture(scope="module")
def shared_vector_store():
    store, docs = initialize_or_load_vector_store()
    return store


def test_vector_store_initialization(shared_vector_store):
    assert shared_vector_store.is_ready is True
    assert shared_vector_store.index.ntotal > 0
    assert len(shared_vector_store.chunks) == shared_vector_store.index.ntotal


def test_retrieval_query_borrowing(shared_vector_store):
    result: RetrievalResult = retrieve_evidence(
        query="borrowing and lending rules",
        vector_store=shared_vector_store,
        top_k=3,
    )
    assert not result.is_empty
    assert len(result.passages) <= 3
    assert result.max_score > 0.3

    # Must retrieve General Code of Conduct Page 2
    top_p = result.passages[0]
    assert "General Code of Conduct" in top_p.document_title
    assert top_p.page_number == 2
    assert "borrowing and lending" in top_p.text.lower()


def test_retrieval_whistleblower_protection(shared_vector_store):
    result: RetrievalResult = retrieve_evidence(
        query="protection of whistleblowers from retaliation",
        vector_store=shared_vector_store,
        top_k=3,
    )
    assert not result.is_empty
    # Must retrieve Whistle Blowing Policy
    titles = [p.document_title for p in result.passages]
    assert "Whistle Blowing Policy" in titles


def test_retrieval_empty_query(shared_vector_store):
    result = retrieve_evidence(query="", vector_store=shared_vector_store)
    assert result.is_empty
    assert len(result.passages) == 0


def test_format_evidence_block():
    c1 = Chunk(
        chunk_id="test_1",
        document_id="doc1",
        document_title="Policy A",
        source_filename="a.pdf",
        page_number=4,
        text="Content of clause 4.",
        token_count_approx=10,
        char_count=20,
        chunk_index=0,
    )
    from src.retrieval import RetrievedPassage
    passages = [RetrievedPassage(chunk=c1, score=0.88, rank=1)]
    block = format_evidence_block(passages)
    assert "Policy A" in block
    assert "Page: 4" in block
    assert "Content of clause 4." in block
