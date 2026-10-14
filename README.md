# Simple RAG Chat

A minimal command-line chatbot built with **LangChain** that uses Retrieval-Augmented Generation (RAG). It finds the most relevant document for your question using embeddings and cosine similarity, then passes it as context to an LLM to generate the answer.

## How It Works

1. **Embed documents**: A small list of documents is converted into vectors with Google's `gemini-embedding-001` model.
2. **Embed the question**: Each user question is embedded the same way.
3. **Retrieve**: Cosine similarity finds the document closest to the question. If the match is below a relevance threshold, no context is returned.
4. **Generate**: The question and retrieved context are sent through a prompt template to `openai/gpt-oss-20b` (served by Groq), which answers only from the given context.

```
User question → Retrieve context → Prompt → LLM → Answer
```

The pipeline is built with LangChain Expression Language (LCEL):

```python
rag_chain = RunnablePassthrough.assign(context=...) | prompt | model | output_parser
```

## Tech Stack

- Python 3.9+
- [LangChain](https://python.langchain.com/)
- Groq (chat model: `openai/gpt-oss-20b`)
- Google Generative AI (embeddings: `gemini-embedding-001`)
- scikit-learn (cosine similarity)
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
pip install langchain langchain-core langchain-groq langchain-google-genai scikit-learn python-dotenv
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

## Usage

```bash
python main.py
```

(Replace `main.py` with your script's filename.)

Example session:

```
RAG Chat created, type 'exit' to quit
You: What is RAG?
AI: RAG stands for Retrieval-Augmented Generation.
You: exit
```

Type `exit` to quit.

## Customizing the Knowledge Base

Edit the `documents` list in the script to add your own content:

```python
documents = [
    "Your first document.",
    "Your second document.",
]
```

The documents are embedded once at startup, so restart the app after changing them.

## Configuration

| Setting | Location | Description |
|---|---|---|
| Chat model | `init_chat_model(...)` | Change the model name or provider |
| Embedding model | `GoogleGenerativeAIEmbeddings(...)` | Change the embedding model |
| `threshhold` | `retrieve_context()` | Minimum similarity score (0 to 1) for a document to count as relevant |

## Project Structure

```
.
├── main.py        # RAG chat application
├── .env           # API keys (not committed)
├── .gitignore
└── README.md
```

## Limitations

- Retrieves only the single best-matching document per question.
- Documents are stored in memory, not in a vector database.
- Intended as a learning example, not for production use.

## Possible Improvements

- Retrieve the top-k documents instead of one
- Load documents from files (PDF, TXT) with a text splitter
- Use a vector store such as FAISS or Chroma
- Add conversation memory

## License

This project is licensed under the MIT License.