import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from local_rag.config import Settings
from local_rag.retrieval import (
    ChromaEvidenceRetriever,
    Evidence,
    RetrievalInput,
    create_retrieval_tool,
)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            embedding_model="embed-model",
            chat_model="chat-model",
            model_provider="ollama",
            dataset_path=Path("dataset.jsonl"),
            database_location=Path("index"),
            collection_name="rag_data",
            retrieval_limit=3,
            relevance_threshold=0.65,
        )

    @patch("local_rag.retrieval.Chroma")
    @patch("local_rag.retrieval.OllamaEmbeddings")
    def test_chroma_retrieval_applies_limits_and_returns_source_metadata(
        self, _embeddings: MagicMock, chroma: MagicMock
    ) -> None:
        store = chroma.return_value
        store.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Supported detail",
                    metadata={"source": "https://example.test", "title": "Example"},
                ),
                0.82,
            ),
            (Document(page_content="No source", metadata={}), 0.75),
        ]

        evidence = ChromaEvidenceRetriever(self.settings).retrieve(
            RetrievalInput(query="What is supported?")
        )

        store.similarity_search_with_relevance_scores.assert_called_once_with(
            "What is supported?", k=3, score_threshold=0.65
        )
        self.assertEqual(
            evidence,
            (Evidence("Supported detail", "https://example.test", "Example", 0.82),),
        )

    def test_retrieval_tool_has_typed_input_and_structured_output(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = (
            Evidence("Fact", "https://source.test", "Source", 0.91),
        )

        retrieval_tool = create_retrieval_tool(retriever)
        result = retrieval_tool.invoke({"query": "Find the fact"})

        self.assertIs(retrieval_tool.args_schema, RetrievalInput)
        self.assertEqual(
            result,
            [
                {
                    "content": "Fact",
                    "source_url": "https://source.test",
                    "title": "Source",
                    "relevance": 0.91,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
