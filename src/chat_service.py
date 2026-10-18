"""
Chat Service Orchestrator for PSW Policy Assistant.
Coordinates multi-turn conversation memory, context-aware query formulation,
evidence retrieval, and grounded Gemini answer generation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import CONVERSATION_HISTORY_WINDOW, DEFAULT_TOP_K
from src.llm import GeminiClient, LLMResponse
from src.retrieval import RetrievalResult, RetrievedPassage, retrieve_evidence
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single chat turn with metadata, citations, and evidence passages."""
    role: str  # "user" or "assistant"
    content: str
    citations: List[str] = field(default_factory=list)
    retrieved_passages: List[RetrievedPassage] = field(default_factory=list)
    retrieval_diagnostics: Optional[Dict[str, any]] = None
    model_used: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    )


class ChatService:
    """Orchestrates RAG workflow for conversational interaction."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: Optional[GeminiClient] = None,
        history_window: int = CONVERSATION_HISTORY_WINDOW,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client or GeminiClient()
        self.history_window = history_window

    def _format_conversation_history(self, history: List[ChatMessage]) -> str:
        """Formats the last N conversation turns as contextual history."""
        if not history:
            return ""

        # Take last window messages
        recent = history[-self.history_window:]
        formatted_turns = []
        for msg in recent:
            speaker = "User" if msg.role == "user" else "Assistant"
            # Truncate very long past responses in context to prevent prompt bloat
            content_snippet = msg.content
            if len(content_snippet) > 400:
                content_snippet = content_snippet[:400] + "..."
            formatted_turns.append(f"{speaker}: {content_snippet}")

        return "\n".join(formatted_turns)

    def _formulate_retrieval_query(self, current_question: str, history: List[ChatMessage]) -> str:
        """
        Enhances retrieval query for short pronoun-heavy follow-up questions
        (e.g., 'What about loans?' or 'Who investigates it?').
        """
        clean_q = current_question.strip()
        words = clean_q.split()

        # If question is very short and there's prior history, blend key terms
        if len(words) <= 5 and history:
            last_user_msgs = [m for m in history if m.role == "user"]
            if last_user_msgs:
                prev_text = last_user_msgs[-1].content
                # Combine prev question context with current follow-up
                combined = f"{prev_text} {clean_q}"
                return combined

        return clean_q

    def ask(
        self,
        question: str,
        history: List[ChatMessage],
        top_k: int = DEFAULT_TOP_K,
        override_api_key: Optional[str] = None,
        override_model: Optional[str] = None,
    ) -> ChatMessage:
        """
        Executes end-to-end RAG turn:
        1. Query formulation from history.
        2. Semantic FAISS retrieval.
        3. Grounded Gemini generation.
        4. Structured citation attachment.
        """
        clean_question = question.strip()
        if not clean_question:
            return ChatMessage(
                role="assistant",
                content="Please ask a question about Pakistan Single Window (PSW) policies.",
            )

        # 1. Update LLM client if user supplied an override key in UI
        if override_api_key and (
            not self.llm_client.is_configured
            or self.llm_client.api_key != override_api_key.strip()
        ):
            self.llm_client = GeminiClient(api_key=override_api_key, model_name=override_model)

        # 2. Formulate search query & retrieve passages
        retrieval_query = self._formulate_retrieval_query(clean_question, history)
        retrieval_result: RetrievalResult = retrieve_evidence(
            query=retrieval_query,
            vector_store=self.vector_store,
            top_k=top_k,
        )

        # 3. Format conversational context
        history_context = self._format_conversation_history(history)

        # 4. Generate grounded answer
        llm_resp: LLMResponse = self.llm_client.generate_grounded_answer(
            question=clean_question,
            passages=retrieval_result.passages,
            conversation_context=history_context,
            override_model=override_model,
        )

        # 5. Extract verified citations from actual retrieved evidence
        citations = retrieval_result.unique_citations

        return ChatMessage(
            role="assistant",
            content=llm_resp.content,
            citations=citations,
            retrieved_passages=retrieval_result.passages,
            retrieval_diagnostics=retrieval_result.get_diagnostic_dict(),
            model_used=llm_resp.model_used,
        )
