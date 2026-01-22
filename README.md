# Local Agentic RAG with Ollama and Chroma DB

A simple, local implementation of Retrieval-Augmented Generation (RAG) using Ollama for embeddings and chat models, and Chroma DB as the vector database. This project enables you to build a chatbot that can answer questions based on your own documents.

## 🎯 Overview

This project consists of two main components:

1. **Embedding Generation** (`1_generate_embedding.py`): Processes JSON-formatted documents, creates embeddings, and stores them in Chroma DB
2. **Chatbot Interface** (`2_start_chatbot.py`): A Streamlit-based web interface that uses an agentic RAG system to answer questions based on the stored documents

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - [Download Python](https://www.python.org/downloads/)
- **Ollama** - [Install Ollama](https://ollama.com/)
  - After installation, pull the required models:
    ```bash
    ollama pull mxbai-embed-large  # Embedding model
    ollama pull llama3.2:3b        # Chat model
    ```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd local-agentic-rag-with-ollama
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root (you can use `.env.example` as a template).

**Note:** Ensure your data file is located at `DATASET_STORAGE_FOLDER/data.txt` in JSONL format (one JSON object per line) with the following structure:
```json
{"url": "https://example.com", "title": "Document Title", "raw_text": "Document content here..."}
```

## 💻 Usage

### Step 1: Generate Embeddings

First, process your documents and create embeddings:

```bash
python 1_generate_embedding.py
```

This script will:
- Load your documents from the configured data file
- Split them into chunks
- Generate embeddings using Ollama
- Store them in Chroma DB

### Step 2: Start the Chatbot

Once embeddings are generated, start the Streamlit chatbot interface:

```bash
streamlit run 2_start_chatbot.py
```

The chatbot will open in your browser. You can now ask questions based on your documents!

## 📁 Project Structure

```
local-agentic-rag-with-ollama/
├── 1_generate_embedding.py    # Script to generate and store embeddings
├── 2_start_chatbot.py         # Streamlit chatbot interface
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── .env.example               # Example environment file
├── chroma_db/                  # Chroma DB storage (created automatically)
└── datasets/                      # Your document data folder
    └── data.txt               # JSONL formatted documents
```

## 🔧 How It Works

1. **Document Processing**: Documents are loaded from JSONL format and split into chunks using `RecursiveCharacterTextSplitter`
2. **Embedding Generation**: Each chunk is converted to embeddings using Ollama's embedding model
3. **Vector Storage**: Embeddings are stored in Chroma DB with metadata (source URL, title)
4. **Retrieval**: When you ask a question, the system retrieves the most relevant document chunks
5. **Generation**: An agentic LLM uses the retrieved context to generate accurate answers with source citations

## 📚 Further Reading

- [Vector Embeddings Explained](https://www.ibm.com/think/topics/vector-embedding)
- [Ollama Embedding Models](https://ollama.com/blog/embedding-models)
- [Chroma DB with LangChain](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [Ollama Text Embeddings](https://python.langchain.com/docs/integrations/text_embedding/ollama/)
- [mxbai-embed-large Model](https://ollama.com/library/mxbai-embed-large)
- [Qwen3 Model](https://ollama.com/library/qwen3)

## ⚠️ Troubleshooting

- **Ollama not found**: Ensure Ollama is installed and running. Check with `ollama list`
- **Model not found**: Pull the required models using `ollama pull <model-name>`
- **Database errors**: Delete the `chroma_db` folder and regenerate embeddings
- **Import errors**: Ensure your virtual environment is activated and all dependencies are installed