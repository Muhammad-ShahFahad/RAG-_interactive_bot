# Load API keys and other environment variables from .env
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from sklearn.metrics.pairwise import cosine_similarity
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


# These are the documents our RAG system will search
documents = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Machine learning allows computers to learn patterns from data.",
    "Python is a programming language commonly used in AI."
]


# Convert all documents into vectors
# We will compare the user's question with these vectors
vectors = embeddings.embed_documents(documents)


# This function finds the document most related to the question
def retrieve_context(question):

    # Convert the user's question into a vector
    question_vector = embeddings.embed_query(question)

    # Compare the question vector with all document vectors
    scores = cosine_similarity(
        [question_vector],
        vectors
    )[0]

    # Get the index of the document with the highest similarity
    best_index = scores.argmax()

    # Get the highest similarity score
    best_score = scores[best_index]

    # If the best document is not similar enough, don't use it
    threshold = 0.5

    if best_score < threshold:
        return "No relevant document found."

    # Return the most relevant document
    return documents[best_index]


# Create the prompt that will be sent to the LLM
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the user's question using the provided context. "
        "If the context does not contain the answer, say you don't know."
    ),
    (
        "human",
        "{question}\n\nContext: {context}"
    )
])


# Keep the original question and add the retrieved context
#
# RunnableLambda runs our retrieve_context() function
# RunnablePassthrough.assign() adds its result as "context"
chain = RunnablePassthrough.assign(
    context=RunnableLambda(
        lambda x: retrieve_context(x["question"])
    )
)


# Complete RAG pipeline:
# retrieve context -> create prompt -> LLM -> convert response to string
rag_chain = chain | prompt | model | output_parser


# Start the chatbot
print("RAG is created, Type 'exit' to quit")

while True:

    # Take question from the user
    user_input = input("You: ")

    # Stop the program when the user types exit
    if user_input.lower() == "exit":
        break

    # Send the question into our RAG chain
    result = rag_chain.invoke({
        "question": user_input
    })

    # Print the final answer
    print(f"AI: {result}")