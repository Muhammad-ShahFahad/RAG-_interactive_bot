# RAG Chat with Chroma

A command-line chatbot built with **LangChain** that uses Retrieval-Augmented Generation (RAG). It retrieves relevant chunks from a persistent **Chroma** vector database using MMR search, then passes them as context to an LLM that answers only from that context.

## How It Works

1. **Load the vector store**: An existing Chroma database (`Chroma_db/`) is loaded from disk, using Google's `gemini-embedding-001` model to embed queries.
2. **Retrieve**: The user's question is embedded, and the retriever uses **MMR (Maximal Marginal Relevance)** to return the top 2 chunks. MMR balances relevance with diversity, so the chunks aren't near-duplicates.
3. **Generate**: The question and retrieved context are sent through a prompt template to `openai/gpt-oss-20b` (served by Groq), which answers only from the given context. If the context has no answer, it says it doesn't know.

```
User question → Chroma retriever (MMR) → Prompt → LLM → Answer
```

The pipeline is built with LangChain Expression Language (LCEL):

```python
rag_chain = RunnablePassthrough.assign(context=...) | prompt | model | output_parser
```

## Tech Stack

- Python 3.9+
- [LangChain](https://python.langchain.com/)
- [Chroma](https://www.trychroma.com/) (persistent vector database)
- Groq (chat model: `openai/gpt-oss-20b`)
- Google Generative AI (embeddings: `gemini-embedding-001`)
- python-dotenv

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain langchain-core langchain-groq langchain-google-genai langchain-chroma langchain-text-splitters python-dotenv
```

### 4. Add your API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

- Get a Groq key at [console.groq.com](https://console.groq.com/)
- Get a Google key at [Google AI Studio](https://aistudio.google.com/)

> **Important:** Never commit your `.env` file. Add it to `.gitignore`.

### 5. Build the vector database

This app **loads an existing** Chroma database from the `Chroma_db/` folder, so it must be created first. Use the same embedding model for indexing and querying. A minimal ingestion script:

```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

texts = ["Your first document...", "Your second document..."]
chunks = splitter.create_documents(texts)

Chroma.from_documents(chunks, embeddings, persist_directory="Chroma_db")
```

## Usage

```bash
python main.py
```

(Replace `main.py` with your script's filename.)

Example session:

```
RAG chat model has started
You: What is RAG?
RAG stands for Retrieval-Augmented Generation.
You: exit
```

Type `exit` to quit.

## Configuration

| Setting | Location | Description |
|---|---|---|
| Chat model | `init_chat_model(...)` | Change the model name or provider |
| Embedding model | `GoogleGenerativeAIEmbeddings(...)` | Must match the model used to build the database |
| `persist_dir` | Top of script | Folder where the Chroma database is stored |
| `search_type` | `as_retriever(...)` | `"mmr"` or `"similarity"` |
| `k` | `search_kwargs` | Number of chunks returned to the LLM |
| `fetch_k` | `search_kwargs` | Number of candidates MMR considers before picking `k` |

## Project Structure

```
.
├── main.py        # RAG chat application
├── Chroma_db/     # Persisted vector database
├── .env           # API keys (not committed)
├── .gitignore
└── README.md
```

## Limitations

- No conversation memory: each question is answered independently.
- Only the top 2 chunks are used as context, so broad questions may miss information.
- The database is read-only from this script; updating the knowledge base requires re-running the ingestion step.

## Possible Improvements

- Add conversation memory and question rewriting for follow-up questions
- Show source documents alongside each answer
- Add a script to ingest PDFs and text files automatically
- Add a web UI (Streamlit or Gradio)

## License

This project is licensed under the MIT License.