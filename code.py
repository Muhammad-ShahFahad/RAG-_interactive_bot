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
# Chat Model
model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)
# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
# Document through which the model take the context
documents = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Machine learning allows computers to learn patterns from data.",
    "Python is a programming language commonly used in AI."
]
# Structured ouput parser
output_parser = StrOutputParser()

# Creating vector representaion of the document
vectors = embeddings.embed_documents(documents)

# Creating a function through which the context will extract form the document
def retrieve_context(question):
    question_vector = embeddings.embed_query(question) # Embedding of the user input
    scores = cosine_similarity(
        [question_vector], # we use the list to convert the question_vecotr in to 2D array, initially it was 1D
        vectors
    )[0] # This calculate the similarities between documents and user input
    best_index = scores.argmax()
    threshhold=0.5 # for the relevance of the answer
    if best_index < threshhold:
        return "No relevent document found"
    
    return documents[best_index]
# Creating prompts for the model
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's question using the provided context. "
     "If the context does not contain the answer, say you don't know."),
    ("human", "{question}\n\nContext: {context}")
])

# Creating chain
'''
in chain we recive the user input in the form of dictionary eg= {"question": "What is RAG?"}
RunnablePassthrough.assign, assign the context to it eg{"question":"....", "context":"..."}
RunnableLambda call the retrieve_context function and give the user_input to it 
'''
chain = RunnablePassthrough.assign(context=RunnableLambda(lambda x: retrieve_context(x["question"])))

# Creating RAG chain
rag_chain = chain | prompt | model | output_parser

# Creaing the Chat
print("RAG Chat created, type 'exit' to quit")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    result = rag_chain.invoke({
        "question":user_input
    })

    print(f"AI: {result}")