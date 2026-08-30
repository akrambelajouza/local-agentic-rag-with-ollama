"""Streamlit presentation for the local RAG chatbot."""

from __future__ import annotations

import logging
from collections.abc import Callable
from urllib.parse import parse_qs, unquote, urlparse

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from local_rag.agent import build_assistant
from local_rag.assistant import Citation, GroundedAssistant
from local_rag.config import Settings, load_settings
from local_rag.pdf_ingestion import (
    PdfIngestionError,
    PdfIngestionSummary,
    PdfUpload,
    ingest_pdf_uploads,
)
from local_rag.readiness import ReadinessReport, assess_readiness

LOGGER = logging.getLogger(__name__)

AssistantProvider = Callable[[Settings], GroundedAssistant]
ReadinessProvider = Callable[[], ReadinessReport]
PdfIngestionProvider = Callable[[Settings, list[PdfUpload]], PdfIngestionSummary]


@st.cache_resource(show_spinner=False)
def get_assistant(settings: Settings) -> GroundedAssistant:
    """Reuse stable model and vector-store clients across Streamlit reruns."""

    return build_assistant(settings)


def render_app(
    *,
    assistant_provider: AssistantProvider | None = None,
    readiness_provider: ReadinessProvider | None = None,
    pdf_ingestion_provider: PdfIngestionProvider | None = None,
    citations_expanded: bool = False,
    chat_enabled: bool = True,
) -> None:
    """Render the local RAG chat application."""

    st.set_page_config(page_title="Local RAG Chatbot", page_icon="📚")
    st.title("📚 Local RAG Chatbot")
    st.markdown(
        "Ask questions grounded in the local document collection. The current demo "
        "covers Python concepts, history, syntax, and common uses."
    )
    st.markdown(
        "**Try asking:** `What is Python?` · `Who created Python?` · "
        "`What are list comprehensions used for?`"
    )

    report = (readiness_provider or assess_readiness)()
    st.subheader("Environment readiness")
    if report.ready:
        st.success("Ready to answer questions locally.")
    else:
        st.warning("Complete the setup actions below before chatting.")
    for check in report.checks:
        icon = "✅" if check.ok else "❌"
        st.write(f"{icon} **{check.label}:** {check.detail}")
        if not check.ok and check.action:
            st.caption(f"Action: {check.action}")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    if not chat_enabled:
        st.info("This deterministic portfolio preview is read-only.")
    elif _render_pdf_ingestion(pdf_ingestion_provider or ingest_pdf_uploads):
        return

    if st.button(
        "Clear conversation",
        disabled=not st.session_state.messages,
        use_container_width=False,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
        return

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                _render_citations(
                    _message_citations(message), expanded=citations_expanded
                )

    submitted_question = st.chat_input(
        "Ask about the indexed documents",
        disabled=(
            not chat_enabled
            or not report.ready
            or st.session_state.pending_question is not None
        ),
    )
    if submitted_question and st.session_state.pending_question is None:
        st.session_state.pending_question = submitted_question
        st.rerun()
        return
    if st.session_state.pending_question is None:
        return

    user_question = st.session_state.pending_question
    if not isinstance(user_question, str) or not user_question.strip():
        st.session_state.pending_question = None
        return

    with st.chat_message("user"):
        st.markdown(user_question)

    try:
        provider = assistant_provider or get_assistant
        assistant = provider(load_settings())
        with st.status(
            "Searching indexed documents and generating an answer…", expanded=True
        ) as progress:
            progress.write("Retrieving relevant evidence and asking the local model.")
            answer = assistant.answer(user_question, st.session_state.messages)
            for event in answer.events:
                progress.write(event.message)
            progress.update(label="Answer ready", state="complete", expanded=False)
    except (ConnectionError, OSError, TimeoutError):
        _show_failure(
            user_question,
            "The local AI service is unavailable right now.",
            "Start Ollama with `ollama serve`, verify the configured models, then retry.",
        )
        return
    except Exception:
        LOGGER.exception("Chat request failed")
        _show_failure(
            user_question,
            "The question could not be completed safely.",
            "Check the local model, vector collection, and retrieval settings, then retry.",
        )
        return
    finally:
        st.session_state.pending_question = None

    with st.chat_message("assistant"):
        st.markdown(answer.text)
        _render_citations(answer.citations, expanded=citations_expanded)
    st.session_state.messages.append(HumanMessage(user_question))
    st.session_state.messages.append(
        AIMessage(
            answer.text,
            additional_kwargs={
                "citations": [citation.to_dict() for citation in answer.citations]
            },
        )
    )


def _render_pdf_ingestion(provider: PdfIngestionProvider) -> bool:
    st.subheader("Add PDF documents")
    st.caption(
        "Upload text-based PDFs to add their pages to the local corpus. "
        "Each file is limited to 20 MB; scanned PDFs require OCR first."
    )
    with st.form("pdf-ingestion", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files stay on this machine and extracted text is stored in datasets/data.txt.",
        )
        submitted = st.form_submit_button("Ingest PDFs", type="primary")
    if submitted is not True:
        return False
    if not uploaded_files:
        st.warning("Select at least one PDF before starting ingestion.")
        return False

    uploads = [PdfUpload(file.name, file.getvalue()) for file in uploaded_files]
    try:
        with st.status("Extracting PDF text and rebuilding the index…", expanded=True):
            summary = provider(load_settings(), uploads)
    except PdfIngestionError as error:
        st.error("The PDF upload could not be ingested.")
        st.caption(str(error))
        return False
    except Exception:
        LOGGER.exception("PDF ingestion failed")
        st.error("The PDF index rebuild failed; the previous corpus remains available.")
        st.caption("Check Ollama and the embedding model, then try the upload again.")
        return False

    st.session_state.messages = []
    st.session_state.pending_question = None
    get_assistant.clear()
    if summary.index is None:
        st.info("These PDF pages were already present; the index was not rebuilt.")
    else:
        file_label = "PDF" if summary.uploaded_file_count == 1 else "PDFs"
        st.success(
            f"Ingested {summary.uploaded_file_count} {file_label} with "
            f"{summary.extracted_page_count} text pages and rebuilt the local index."
        )
    st.rerun()
    return True


def _message_citations(message: AIMessage) -> tuple[Citation, ...]:
    stored = message.additional_kwargs.get("citations", [])
    citations = (Citation.from_dict(item) for item in stored)
    return tuple(citation for citation in citations if citation is not None)


def _show_failure(question: str, message: str, guidance: str) -> None:
    st.error(message)
    st.caption(guidance)
    st.session_state.messages.append(HumanMessage(question))
    st.session_state.messages.append(AIMessage(message))


def _render_citations(
    citations: tuple[Citation, ...], *, expanded: bool = False
) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        with st.expander(citation.title, expanded=expanded):
            if citation.url.startswith("local-pdf://"):
                st.caption(_local_pdf_source_label(citation.url))
            else:
                st.markdown(f"[{citation.url}]({citation.url})")
            if citation.excerpt:
                st.caption(citation.excerpt)


def _local_pdf_source_label(url: str) -> str:
    parsed = urlparse(url)
    filename = unquote(parsed.path.rsplit("/", 1)[-1]) or "uploaded PDF"
    page = parse_qs(parsed.fragment).get("page", [""])[0]
    suffix = f" · page {page}" if page else ""
    return f"Local PDF: {filename}{suffix}"
