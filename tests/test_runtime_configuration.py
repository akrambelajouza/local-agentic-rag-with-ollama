import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_rag.agent import build_agent_executor
from local_rag.config import Settings
from local_rag.ingestion import CorpusDocument, generate_embeddings


class RuntimeConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            embedding_model="embed-model",
            chat_model="chat-model",
            model_provider="ollama",
            dataset_path=Path("dataset.jsonl"),
            database_location=Path("missing-index"),
            collection_name="rag_data",
            ollama_base_url="http://ollama.internal:11434",
        )

    @patch("local_rag.agent.AgentExecutor")
    @patch("local_rag.agent.create_tool_calling_agent")
    @patch("local_rag.agent.init_chat_model")
    @patch("local_rag.agent.Chroma")
    @patch("local_rag.agent.OllamaEmbeddings")
    def test_chat_clients_use_configured_ollama_url(
        self,
        embeddings: MagicMock,
        _chroma: MagicMock,
        chat_model: MagicMock,
        _create_agent: MagicMock,
        _executor: MagicMock,
    ) -> None:
        build_agent_executor(self.settings)

        embeddings.assert_called_once_with(
            model="embed-model", base_url="http://ollama.internal:11434"
        )
        chat_model.assert_called_once_with(
            "chat-model",
            model_provider="ollama",
            temperature=0,
            base_url="http://ollama.internal:11434",
        )

    @patch("local_rag.ingestion.RecursiveCharacterTextSplitter")
    @patch("local_rag.ingestion.Chroma")
    @patch("local_rag.ingestion.OllamaEmbeddings")
    @patch("local_rag.ingestion.load_documents")
    def test_ingestion_client_uses_configured_ollama_url(
        self,
        load_documents: MagicMock,
        embeddings: MagicMock,
        _chroma: MagicMock,
        splitter: MagicMock,
    ) -> None:
        load_documents.return_value = [CorpusDocument("url", "title", "text")]
        splitter.return_value.create_documents.return_value = []

        generate_embeddings(self.settings)

        embeddings.assert_called_once_with(
            model="embed-model", base_url="http://ollama.internal:11434"
        )


if __name__ == "__main__":
    unittest.main()
