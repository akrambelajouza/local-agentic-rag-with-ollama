"""Streamlit presentation for the local RAG chatbot."""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from local_rag.agent import build_agent_executor
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

    user_question = st.chat_input(
        "Ask about the indexed documents", disabled=not report.ready
    )
    if not user_question:
        return

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    executor = build_agent_executor(load_settings())
    result = executor.invoke(
        {"input": user_question, "chat_history": st.session_state.messages}
    )
    ai_message = result["output"]

    with st.chat_message("assistant"):
        st.markdown(ai_message)
    st.session_state.messages.append(AIMessage(ai_message))
