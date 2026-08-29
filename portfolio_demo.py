"""Render a deterministic portfolio preview through the production UI."""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from local_rag.app import render_app
from local_rag.assistant import Citation
from local_rag.readiness import ReadinessCheck, ReadinessReport

QUESTION = "Who created Python?"
ANSWER = (
    "Python was created by Guido van Rossum while he was working at CWI in the "
    "Netherlands."
)
CITATION = Citation(
    "Who Created Python?",
    "https://www.python.org/about/help/",
    "Python was created by Guido van Rossum, who developed it while working at CWI.",
)


def demo_readiness() -> ReadinessReport:
    return ReadinessReport(
        (
            ReadinessCheck(
                "demo",
                "Portfolio preview",
                True,
                "Deterministic answer from the included sample corpus.",
            ),
        )
    )


if "portfolio_demo_loaded" not in st.session_state:
    st.session_state.messages = [
        HumanMessage(QUESTION),
        AIMessage(
            ANSWER,
            additional_kwargs={"citations": [CITATION.to_dict()]},
        ),
    ]
    st.session_state.pending_question = None
    st.session_state.portfolio_demo_loaded = True

render_app(
    readiness_provider=demo_readiness,
    citations_expanded=True,
    chat_enabled=False,
)
