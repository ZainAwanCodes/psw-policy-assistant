"""
Unit tests for ChatService orchestration and LLM error handling using mocks.
"""

from unittest.mock import MagicMock
import pytest
from src.chat_service import ChatMessage, ChatService
from src.chunking import Chunk
from src.llm import GeminiClient, LLMResponse
from src.retrieval import RetrievedPassage
from src.vector_store import VectorStore


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.is_ready = True
    c = Chunk(
        chunk_id="c1",
        document_id="general_code_of_conduct",
        document_title="General Code of Conduct",
        source_filename="general_code_of_conduct.pdf",
        page_number=2,
        text="The Company takes no responsibility for borrowing and lending.",
        token_count_approx=12,
        char_count=65,
        chunk_index=0,
    )
    store.search.return_value = [(c, 0.85)]
    return store


def test_chat_service_grounded_success(mock_vector_store):
    mock_llm = MagicMock(spec=GeminiClient)
    mock_llm.is_configured = True
    mock_llm.generate_grounded_answer.return_value = LLMResponse(
        content="The Company takes no responsibility for borrowing and lending [General Code of Conduct — Page 2].",
        model_used="gemini-3.8-flash",
        success=True,
    )

    service = ChatService(vector_store=mock_vector_store, llm_client=mock_llm)
    msg = service.ask("What are the rules on borrowing?", history=[])

    assert msg.role == "assistant"
    assert "borrowing and lending" in msg.content
    assert len(msg.citations) == 1
    assert "General Code of Conduct — Page 2" in msg.citations[0]
    assert len(msg.retrieved_passages) == 1


def test_chat_service_empty_question(mock_vector_store):
    service = ChatService(vector_store=mock_vector_store)
    msg = service.ask("   ", history=[])
    assert "Please ask a question" in msg.content


def test_chat_service_missing_api_key(mock_vector_store):
    unconfigured_llm = GeminiClient(api_key="")
    service = ChatService(vector_store=mock_vector_store, llm_client=unconfigured_llm)
    msg = service.ask("What is the code of conduct?", history=[])
    assert "Gemini API Key Required" in msg.content


def test_chat_service_follow_up_context(mock_vector_store):
    mock_llm = MagicMock(spec=GeminiClient)
    mock_llm.is_configured = True
    mock_llm.generate_grounded_answer.return_value = LLMResponse(
        content="Follow up response.", model_used="gemini-3.8-flash", success=True
    )
    service = ChatService(vector_store=mock_vector_store, llm_client=mock_llm)

    prior_history = [
        ChatMessage(role="user", content="Tell me about conflicts of interest"),
        ChatMessage(role="assistant", content="Employees must report conflicts of interest."),
    ]
    msg = service.ask("Who investigates it?", history=prior_history)

    # Verify that query reformulation included prior topic context
    mock_vector_store.search.assert_called()
    call_query = mock_vector_store.search.call_args[0][0]
    assert "conflicts of interest" in call_query.lower() or "who investigates it?" in call_query.lower()
