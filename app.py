"""
PSW Policy Assistant - Streamlit Application.
A production-grade, document-grounded RAG chatbot for exploring
Pakistan Single Window (PSW) policy documents with page-level citations.
"""

import os
from typing import Dict, List, Optional
import streamlit as st

import config
from src.chat_service import ChatMessage, ChatService
from src.embeddings import EmbeddingEngine
from src.ingestion import Document, ingest_all_policies
from src.llm import GeminiClient
from src.vector_store import VectorStore, initialize_or_load_vector_store

# Set page configuration
st.set_page_config(
    page_title="PSW Policy Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern enterprise aesthetic
st.markdown(
    """
    <style>
    /* Global layout & typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        max-width: 960px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        margin: 0 auto;
    }

    /* Header styling */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        border-bottom: 1px solid #E7E5E4;
        margin-bottom: 1.5rem;
    }
    .app-title-group {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .app-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1C1917;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .badge-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-ready {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    /* Hero headline */
    .hero-section {
        margin-bottom: 1.75rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1C1917;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #57534E;
        line-height: 1.5;
        margin: 0;
    }

    /* Citation badge */
    .citation-tag {
        display: inline-block;
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Evidence card */
    .evidence-card {
        background: #FAFAF9;
        border: 1px solid #E7E5E4;
        border-left: 3px solid #0F766E;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        color: #292524;
    }
    .evidence-meta {
        font-weight: 600;
        color: #0F766E;
        margin-bottom: 0.35rem;
        font-size: 0.8rem;
    }
    .evidence-body {
        color: #44403C;
        white-space: pre-wrap;
        line-height: 1.45;
    }

    /* Suggested query chip styling */
    .suggestion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }

    /* Sidebar cards */
    .sidebar-doc-card {
        padding: 0.65rem 0.85rem;
        background: #FFFFFF;
        border: 1px solid #E7E5E4;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .doc-name {
        font-weight: 600;
        font-size: 0.85rem;
        color: #1C1917;
    }
    .doc-stat {
        font-size: 0.75rem;
        color: #78716C;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing RAG vector store and embedding engine...")
def load_rag_components() -> tuple[VectorStore, Dict[str, Document], EmbeddingEngine]:
    """Caches in-memory embedding model and vector index."""
    embed_engine = EmbeddingEngine.get_instance()
    docs = ingest_all_policies()
    store, docs = initialize_or_load_vector_store(
        docs=docs,
        force_rebuild=False,
        embedding_engine=embed_engine,
    )
    return store, docs, embed_engine


# Initialize components
vector_store, policy_docs, embedding_engine = load_rag_components()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==========================================
# SIDEBAR: Status, Config & Diagnostics
# ==========================================
with st.sidebar:
    st.markdown("### 🏛️ Knowledge Base Status")
    st.caption("Indexed Pakistan Single Window Policies:")

    indexable_count = 0
    total_pages_indexed = 0

    for doc_id, doc in policy_docs.items():
        if doc.is_indexable:
            indexable_count += 1
            total_pages_indexed += doc.total_pages
            status_icon = "🟢"
            status_desc = f"{doc.total_pages} pages | {doc.total_chars:,} chars"
        elif doc.is_scanned:
            status_icon = "🟡"
            status_desc = "Scanned image PDF (OCR required)"
        else:
            status_icon = "🔴"
            status_desc = doc.status

        st.markdown(
            f"""
            <div class="sidebar-doc-card">
                <div class="doc-name">{status_icon} {doc.title}</div>
                <div class="doc-stat">{status_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if any(d.is_scanned for d in policy_docs.values()):
        with st.expander("ℹ️ Why are some policies marked OCR required?"):
            st.info(
                "Documents like 'Harassment Policy' and 'Anti Corruption Policy' are photocopied scans "
                "from Konica Minolta scanners without digital text layers. To prevent silent hallucinations, "
                "the system detects 0 extractable characters and requires an OCR pipeline or digital copy to index them."
            )

    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")

    # API Key Input — never pre-fill the actual key into the UI
    env_key_configured = bool(config.get_gemini_api_key())
    api_key_override = st.text_input(
        "Gemini API Key",
        value="",
        type="password",
        placeholder="Loaded from .env" if env_key_configured else "Paste your Gemini API key",
        help="Your key is read from .env automatically. Only enter a key here to override it for this session.",
    )
    if env_key_configured and not api_key_override.strip():
        st.caption("🔑 API key loaded from environment.")

    # Resolve effective API key: sidebar override > .env
    effective_api_key = api_key_override.strip() or config.get_gemini_api_key() or ""

    # Model Selection
    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
        index=0,
        help="Configured Google GenAI model. gemini-3.8-flash is the recommended default.",
    )

    # Retrieval Top-K
    top_k_val = st.slider("Retrieval Passages (Top-K)", min_value=2, max_value=8, value=config.DEFAULT_TOP_K)

    # Diagnostics Toggle
    show_diagnostics = st.toggle("Show Retrieval Diagnostics", value=False)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding FAISS index from documents..."):
                docs = ingest_all_policies()
                vector_store, policy_docs = initialize_or_load_vector_store(
                    docs=docs,
                    force_rebuild=True,
                    embedding_engine=embedding_engine,
                )
                st.success("Index rebuilt!")
                st.rerun()

    with col_btn2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

    st.markdown("---")
    st.caption(
        f"**Model:** `{embedding_engine.model_name}`\n\n"
        f"**Vectors:** `{vector_store.index.ntotal if vector_store.index else 0}` in FAISS IndexFlatIP\n\n"
        "🔒 Evidence-grounded generation. User secrets are never logged."
    )


# ==========================================
# MAIN INTERFACE
# ==========================================

# 1. Compact Application Header
api_configured = bool(effective_api_key)
status_class = "badge-ready" if (vector_store.is_ready and api_configured) else "badge-warning"
status_text = "Ready & Grounded" if (vector_store.is_ready and api_configured) else "API Key Needed"

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title-group">
            <span style="font-size: 1.5rem;">⚖️</span>
            <span class="app-title">PSW Policy Assistant</span>
        </div>
        <div class="badge-status {status_class}">
            ● {status_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Hero Headline & Subtitle
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">Your policies, answered with evidence.</div>
        <div class="hero-subtitle">
            Explore Pakistan Single Window (PSW) policies in natural language.
            Every answer is verified against indexed document passages with exact page citations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. Suggested Question Chips (Only display when chat is empty)
if not st.session_state.messages:
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #78716C; margin-bottom: 0.5rem;'>SUGGESTED INQUIRIES</p>", unsafe_allow_html=True)
    suggestions = [
        "What are the rules regarding borrowing and lending?",
        "How does PSW protect whistleblowers from retaliation?",
        "Can an employee undertake outside trade or employment?",
        "How long must records of whistleblowing investigations be kept?",
        "What are the provisions of the Harassment Policy?",
    ]

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s4, col_s5 = st.columns(2)

    if col_s1.button(suggestions[0], use_container_width=True):
        st.session_state.pending_prompt = suggestions[0]
        st.rerun()
    if col_s2.button(suggestions[1], use_container_width=True):
        st.session_state.pending_prompt = suggestions[1]
        st.rerun()
    if col_s3.button(suggestions[2], use_container_width=True):
        st.session_state.pending_prompt = suggestions[2]
        st.rerun()
    if col_s4.button(suggestions[3], use_container_width=True):
        st.session_state.pending_prompt = suggestions[3]
        st.rerun()
    if col_s5.button(suggestions[4], use_container_width=True):
        st.session_state.pending_prompt = suggestions[4]
        st.rerun()

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)


# 4. Render Conversation Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display structured citations if present
        if msg["role"] == "assistant" and msg.get("citations"):
            st.markdown("<div style='margin-top: 0.75rem; margin-bottom: 0.4rem; font-size: 0.8rem; font-weight: 600; color: #57534E;'>SUPPORTING SOURCES</div>", unsafe_allow_html=True)
            citation_html = "".join([f"<span class='citation-tag'>📄 {c}</span>" for c in msg["citations"]])
            st.markdown(citation_html, unsafe_allow_html=True)

        # Expandable Supporting Evidence Passages
        if msg["role"] == "assistant" and msg.get("passages"):
            with st.expander(f"🔎 View Supporting Evidence ({len(msg['passages'])} retrieved passages)"):
                for idx, p in enumerate(msg["passages"], 1):
                    score_val = p.get("score", 0.0)
                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <div class="evidence-meta">
                                Passage {idx} | {p['document_title']} — Page {p['page_number']} (Score: {score_val:.3f})
                            </div>
                            <div class="evidence-body">{p['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Diagnostics View
        if msg["role"] == "assistant" and show_diagnostics and msg.get("diagnostics"):
            with st.expander("🛠️ Retrieval Diagnostics (Dev Mode)"):
                st.json(msg["diagnostics"])


# 5. Handle Prompt Input (Chat Composer & Suggested Chips)
user_prompt = st.chat_input("Ask a question about PSW policies...")

# If a suggested question chip was clicked, process it
if st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_prompt:
    clean_user_prompt = user_prompt.strip()

    # Append user turn
    st.session_state.messages.append({
        "role": "user",
        "content": clean_user_prompt,
    })

    # Display user bubble immediately
    with st.chat_message("user"):
        st.markdown(clean_user_prompt)

    # Generate assistant turn
    with st.chat_message("assistant"):
        with st.spinner("Searching indexed policies & verifying evidence..."):
            llm_client = GeminiClient(
                api_key=effective_api_key,
                model_name=selected_model,
            )
            chat_service = ChatService(
                vector_store=vector_store,
                llm_client=llm_client,
                history_window=config.CONVERSATION_HISTORY_WINDOW,
            )

            # Reconstruct ChatMessage objects for history
            history_objs = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in st.session_state.messages[:-1]
            ]

            resp_msg: ChatMessage = chat_service.ask(
                question=clean_user_prompt,
                history=history_objs,
                top_k=top_k_val,
                override_api_key=effective_api_key,
                override_model=selected_model,
            )

            # Display response content
            st.markdown(resp_msg.content)

            # Render citation tags
            if resp_msg.citations:
                st.markdown("<div style='margin-top: 0.75rem; margin-bottom: 0.4rem; font-size: 0.8rem; font-weight: 600; color: #57534E;'>SUPPORTING SOURCES</div>", unsafe_allow_html=True)
                citation_html = "".join([f"<span class='citation-tag'>📄 {c}</span>" for c in resp_msg.citations])
                st.markdown(citation_html, unsafe_allow_html=True)

            # Render evidence accordion
            if resp_msg.retrieved_passages:
                with st.expander(f"🔎 View Supporting Evidence ({len(resp_msg.retrieved_passages)} retrieved passages)"):
                    for idx, p in enumerate(resp_msg.retrieved_passages, 1):
                        st.markdown(
                            f"""
                            <div class="evidence-card">
                                <div class="evidence-meta">
                                    Passage {idx} | {p.document_title} — Page {p.page_number} (Score: {p.score:.3f})
                                </div>
                                <div class="evidence-body">{p.text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Render diagnostics if enabled
            if show_diagnostics and resp_msg.retrieval_diagnostics:
                with st.expander("🛠️ Retrieval Diagnostics (Dev Mode)"):
                    st.json(resp_msg.retrieval_diagnostics)

            # Save assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": resp_msg.content,
                "citations": resp_msg.citations,
                "passages": [
                    {
                        "document_title": p.document_title,
                        "page_number": p.page_number,
                        "score": p.score,
                        "text": p.text,
                    }
                    for p in resp_msg.retrieved_passages
                ],
                "diagnostics": resp_msg.retrieval_diagnostics,
                "model_used": resp_msg.model_used,
            })
