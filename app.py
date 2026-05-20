import os
import certifi

# Fix broken SSL_CERT_FILE path that conda may set to a non-existent file
_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not os.path.isfile(_ssl_cert):
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings and connect to existing Pinecone index
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Build the RAG chain using LCEL
chatModel = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | chatModel
    | StrOutputParser()
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)
    response = rag_chain.invoke(msg)
    print("Bot:", response)
    return str(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
