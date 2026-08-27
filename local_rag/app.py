"""Streamlit presentation for the local RAG chatbot."""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from local_rag.agent import build_assistant
from local_rag.assistant import Citation
from local_rag.config import load_settings
from local_rag.readiness import assess_readiness


def render_app() -> None:
    """Render the local RAG chat application."""

    st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
    st.title("🦜 Agentic RAG Chatbot")

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

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                _render_citations(_message_citations(message))

    user_question = st.chat_input(
        "Ask about the indexed documents", disabled=not report.ready
    )
    if not user_question:
        return

    with st.chat_message("user"):
        st.markdown(user_question)

    assistant = build_assistant(load_settings())
    answer = assistant.answer(user_question, st.session_state.messages)
    ai_message = answer.text

    with st.chat_message("assistant"):
        st.markdown(ai_message)
        _render_citations(answer.citations)
    st.session_state.messages.append(HumanMessage(user_question))
    st.session_state.messages.append(
        AIMessage(
            ai_message,
            additional_kwargs={
                "citations": [
                    {"title": citation.title, "url": citation.url}
                    for citation in answer.citations
                ]
            },
        )
    )


def _message_citations(message: AIMessage) -> tuple[Citation, ...]:
    stored = message.additional_kwargs.get("citations", [])
    return tuple(
        Citation(title=str(item["title"]), url=str(item["url"]))
        for item in stored
        if isinstance(item, dict) and "title" in item and "url" in item
    )


def _render_citations(citations: tuple[Citation, ...]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        st.markdown(f"- [{citation.title}]({citation.url})")
