"""Agent construction and retrieval adaptation for the local RAG runtime."""

from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_ollama import OllamaEmbeddings

from local_rag.config import Settings


AGENT_PROMPT = PromptTemplate.from_template(
    """
You are a helpful assistant. You will be provided with a query and a chat history.
Your task is to retrieve relevant information from the vector store and provide a response.
For this you use the tool 'retrieve' to get the relevant information.

The query is as follows:
{input}

The chat history is as follows:
{chat_history}

Please provide a concise and informative response based on the retrieved information.
If you don't know the answer, say "I don't know" (and don't provide a source).

You can use the scratchpad to store any intermediate results or notes.
The scratchpad is as follows:
{agent_scratchpad}

For every piece of information you provide, also provide the source.

Return text as follows:

<Answer to the question>
Source: source_url
"""
)


def build_agent_executor(settings: Settings) -> AgentExecutor:
    """Create the current Ollama, Chroma, retrieval-tool agent at runtime."""

    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.database_location),
    )
    llm = init_chat_model(
        settings.chat_model,
        model_provider=settings.model_provider,
        temperature=0,
        base_url=settings.ollama_base_url,
    )
    retrieve = _create_retrieval_tool(vector_store)
    tools = [retrieve]
    agent = create_tool_calling_agent(llm, tools, AGENT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def _create_retrieval_tool(vector_store: Chroma) -> BaseTool:
    @tool
    def retrieve(query: Any) -> str:
        """Retrieve information related to a query."""

        normalized_query = _normalize_query(query)
        retrieved_documents = vector_store.similarity_search(normalized_query, k=2)
        return "".join(
            f"Source: {document.metadata['source']}\n"
            f"Content: {document.page_content}\n\n"
            for document in retrieved_documents
        )

    return retrieve


def _normalize_query(query: Any) -> str:
    if not isinstance(query, dict):
        return str(query)
    if "query" in query and isinstance(query["query"], dict):
        return str(query["query"].get("value", query["query"]))
    for key in ("value", "query", "input", "text"):
        if key in query:
            return str(query[key])
    return next(
        (str(value) for value in query.values() if isinstance(value, str) and value),
        str(query),
    )
