# Load API keys and other environment variables from .env
from dotenv import load_dotenv
load_dotenv(override=True)
import os

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import re


# Chat model that will generate the final answer
model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)


# Embedding model used to convert text into vectors
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# Converts the model's response into a normal string
output_parser = StrOutputParser()

# Path of existing database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(BASE_DIR, "Chroma_db")

# Load the existing database
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings,
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":2},
    fetch_k = 5

)



# Prompts
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the user's question using only the provided context. "
        "If the context does not contain the answer, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])

# Chain for retrieving context
chain = RunnablePassthrough.assign(context=RunnableLambda(lambda x: "\n\n".join(
    doc.page_content for doc in retriever.invoke(x["question"])
)))

# RAG chain
rag_chain = chain | prompt | model | output_parser

# Chat Model
print("RAG chat model has started")

while True:
    user_input = input("You: ")
    if user_input.lower()=='exit':
        break
    result = rag_chain.invoke({
        "question": user_input
    })

    print(result)