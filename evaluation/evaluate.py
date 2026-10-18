"""
RAG Evaluation Harness for PSW Policy Assistant.
Evaluates:
1. Retrieval Hit@K (evidence retrieval accuracy).
2. Citation validity (verifies citations match retrieved metadata).
3. Abstention accuracy on out-of-domain, scanned, and adversarial queries.
4. Optional live LLM answer generation and qualitative evaluation.
"""

import argparse
import io
import json
import logging
import sys
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force utf-8 stdout on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import DEFAULT_TOP_K
from src.chat_service import ChatService
from src.llm import GeminiClient
from src.retrieval import RetrievalResult, retrieve_evidence
from src.vector_store import initialize_or_load_vector_store

logger = logging.getLogger(__name__)


def load_evaluation_dataset(path: Optional[Path] = None) -> List[Dict[str, any]]:
    """Loads evaluation test dataset from JSON file."""
    if path is None:
        path = Path(__file__).parent / "evaluation_questions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_rag(with_llm: bool = False, top_k: int = DEFAULT_TOP_K) -> Dict[str, any]:
    """
    Runs systematic evaluation across the benchmark dataset.
    """
    dataset = load_evaluation_dataset()
    vector_store, docs = initialize_or_load_vector_store()
    chat_service = ChatService(vector_store=vector_store) if with_llm else None

    total_questions = len(dataset)
    answerable_count = sum(1 for q in dataset if q.get("is_answerable"))
    unanswerable_count = total_questions - answerable_count

    retrieval_hits = 0
    abstention_successes = 0
    citation_checks_passed = 0
    results_detail = []

    print(f"\n==================================================")
    print(f"  PSW Policy Assistant - RAG Pipeline Evaluation")
    print(f"==================================================")
    print(f"Total benchmark questions: {total_questions}")
    print(f"Answerable policy questions: {answerable_count}")
    print(f"Unanswerable / Scanned / Adversarial questions: {unanswerable_count}")
    print(f"Retrieval Top-K: {top_k}")
    print(f"Live LLM Evaluation: {'ENABLED' if with_llm else 'DISABLED (deterministic retrieval mode)'}\n")

    for idx, item in enumerate(dataset, 1):
        q_id = item["id"]
        question = item["question"]
        is_answerable = item["is_answerable"]
        exp_doc = item.get("expected_document")
        exp_page = item.get("expected_page")
        category = item.get("category")

        # 1. Retrieval
        retrieval: RetrievalResult = retrieve_evidence(query=question, vector_store=vector_store, top_k=top_k)

        # Check Hit@K for answerable questions
        hit = False
        if is_answerable and exp_doc:
            for p in retrieval.passages:
                doc_match = exp_doc.lower() in p.document_title.lower()
                page_match = (exp_page is None) or (p.page_number == exp_page)
                if doc_match and page_match:
                    hit = True
                    break
            if hit:
                retrieval_hits += 1

        # 2. Optional Live Generation
        llm_answer = None
        abstained = False
        citations_valid = True

        if with_llm and chat_service:
            msg = chat_service.ask(question=question, history=[], top_k=top_k)
            llm_answer = msg.content

            # Check abstention on unanswerable questions
            abstain_phrases = [
                "do not contain sufficient evidence",
                "do not provide enough information",
                "does not provide sufficient information",
                "insufficient information",
                "not enough information",
                "scanned",
                "ocr",
            ]
            if not is_answerable:
                abstained = any(phrase in llm_answer.lower() for phrase in abstain_phrases)
                if abstained:
                    abstention_successes += 1

            # Validate citations exist in retrieved passages
            retrieved_labels = set(retrieval.unique_citations)
            for cited in msg.citations:
                if cited not in retrieved_labels:
                    citations_valid = False
            if citations_valid:
                citation_checks_passed += 1

        status_icon = "✅" if (hit if is_answerable else (abstained if with_llm else True)) else "❌"
        print(f"[{status_icon}] Q{idx} ({category}): {question[:65]}...")
        if is_answerable:
            print(f"     Expected: {exp_doc} (Page {exp_page}) | Hit@{top_k}: {hit}")
        else:
            print(f"     Expected: Abstention / Guardrail | Abstained: {abstained if with_llm else 'N/A (deterministic)'}")

        results_detail.append({
            "id": q_id,
            "category": category,
            "question": question,
            "is_answerable": is_answerable,
            "hit_at_k": hit if is_answerable else None,
            "abstained": abstained if with_llm else None,
            "max_score": retrieval.max_score,
            "citations_valid": citations_valid if with_llm else None,
        })

    # Summary Metrics
    hit_rate = (retrieval_hits / answerable_count) * 100 if answerable_count else 0.0
    abstention_rate = (abstention_successes / unanswerable_count) * 100 if (with_llm and unanswerable_count) else None

    print(f"\n---------------- SUMMARY METRICS ----------------")
    print(f"Retrieval Hit@{top_k} (Answerable questions): {retrieval_hits}/{answerable_count} ({hit_rate:.1f}%)")
    if with_llm:
        print(f"Abstention Accuracy (Unanswerable/Scanned): {abstention_successes}/{unanswerable_count} ({abstention_rate:.1f}%)")
        print(f"Citation Grounding Accuracy: {citation_checks_passed}/{total_questions} ({citation_checks_passed/total_questions*100:.1f}%)")
    print(f"-------------------------------------------------\n")

    return {
        "total_questions": total_questions,
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "retrieval_hits": retrieval_hits,
        "hit_rate_percent": round(hit_rate, 2),
        "abstention_rate_percent": round(abstention_rate, 2) if abstention_rate is not None else None,
        "results": results_detail,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PSW Policy Assistant RAG pipeline.")
    parser.add_argument("--with-llm", action="store_true", help="Run live Gemini LLM answer generation and judge.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Retrieval Top-K passages.")
    args = parser.parse_args()

    evaluate_rag(with_llm=args.with_llm, top_k=args.top_k)
