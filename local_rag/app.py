"""Streamlit presentation for the local RAG chatbot."""

from __future__ import annotations

from collections.abc import Callable
import logging

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from local_rag.agent import build_assistant
from local_rag.assistant import Citation, GroundedAssistant
from local_rag.config import Settings, load_settings
from local_rag.readiness import assess_readiness


LOGGER = logging.getLogger(__name__)

AssistantProvider = Callable[[Settings], GroundedAssistant]


@st.cache_resource(show_spinner=False)
def get_assistant(settings: Settings) -> GroundedAssistant:
    """Reuse stable model and vector-store clients across Streamlit reruns."""

    return build_assistant(settings)


def render_app(*, assistant_provider: AssistantProvider | None = None) -> None:
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

    report = assess_readiness()
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
                _render_citations(_message_citations(message))

    submitted_question = st.chat_input(
        "Ask about the indexed documents",
        disabled=not report.ready or st.session_state.pending_question is not None,
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
        _render_citations(answer.citations)
    st.session_state.messages.append(HumanMessage(user_question))
    st.session_state.messages.append(
        AIMessage(
            answer.text,
            additional_kwargs={
                "citations": [citation.to_dict() for citation in answer.citations]
            },
        )
    )


def _message_citations(message: AIMessage) -> tuple[Citation, ...]:
    stored = message.additional_kwargs.get("citations", [])
    citations = (Citation.from_dict(item) for item in stored)
    return tuple(citation for citation in citations if citation is not None)


def _show_failure(question: str, message: str, guidance: str) -> None:
    st.error(message)
    st.caption(guidance)
    st.session_state.messages.append(HumanMessage(question))
    st.session_state.messages.append(AIMessage(message))


def _render_citations(citations: tuple[Citation, ...]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        with st.expander(citation.title):
            st.markdown(f"[{citation.url}]({citation.url})")
            if citation.excerpt:
                st.caption(citation.excerpt)
