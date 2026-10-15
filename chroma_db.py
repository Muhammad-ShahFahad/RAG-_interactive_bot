# We donot want to create the database everytime we run the code 
# For this we have created a seperate database which we call in our code.py

from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding model used to convert text into vectors
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
) 


# Text to split
text = """
Retrieval-Augmented Generation (RAG) is a technique used in
Generative AI. It retrieves relevant information from documents
and provides that information to the language model.

Machine learning allows computers to learn patterns from data.
It is commonly used for prediction, classification, and many
other tasks.

Python is a popular programming language used in AI and machine
learning because it has many useful libraries.
"""

# Dividing the document in to the chunks
chunks = splitter.split_text(text)

# Vector of chunks
vectors = embeddings.embed_documents(chunks)

# Chroma db use to store embeddings

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="Chroma_db"
)