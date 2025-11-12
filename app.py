# # import langchain_core
# # setattr(langchain_core, "memory", None)  # temporary patch for old modules

# # from flask import Flask, render_template, jsonify, request
# # from src.helper import downloade_embeddings
# # from langchain_pinecone import PineconeVectorStore
# # from langchain_groq import ChatGroq
# # # from langchain.chains import create_retrieval_chain
# # # from langchain.chains.combine_documents import create_stuff_documents_chain  # ✅ fixed import
# # from langchain_classic.chains import create_retrieval_chain
# # from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# # from langchain_core.prompts import ChatPromptTemplate
# # from dotenv import load_dotenv
# # from src.prompt import *
# # import os

# # app = Flask(__name__)
# # load_dotenv()

# # PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# # GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# # os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
# # os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

# # embeddings = downloade_embeddings()

# # index_name = "medical-chatbot"

# # # ✅ Load Pinecone index
# # docsearch = PineconeVectorStore.from_existing_index(
# #     index_name=index_name,
# #     embedding=embeddings
# # )

# # retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# # chatModel = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# # prompt = ChatPromptTemplate.from_messages([
# #     ("system", system_prompt),
# #     ("human", "{input}")
# # ])

# # question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# # rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # @app.route("/")
# # def index():
# #     return render_template("chat.html")

# # @app.route("/get", methods=["POST"])
# # def chat():
# #     msg = request.form["msg"]
# #     print("User:", msg)
# #     response = rag_chain.invoke({"input": msg})
# #     print("Response:", response["answer"])
# #     return str(response["answer"])

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=8080)










# import langchain_core
# setattr(langchain_core, "memory", None)  # temporary patch for old modules

# from flask import Flask, render_template, request
# from langchain_pinecone import PineconeVectorStore
# from langchain_groq import ChatGroq
# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.helper import downloade_embeddings
# from src.prompt import *
# import os

# # ---------------------------------------------------------
# # Flask setup
# # ---------------------------------------------------------
# app = Flask(__name__)
# load_dotenv()

# # ---------------------------------------------------------
# # Environment variables
# # ---------------------------------------------------------
# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# # ---------------------------------------------------------
# # Global variables (lazy-loaded objects)
# # ---------------------------------------------------------
# embeddings = None
# retriever = None
# rag_chain = None

# # ---------------------------------------------------------
# # Lazy initialization function
# # ---------------------------------------------------------
# def init_rag_chain():
#     """Initialize embeddings, retriever, and RAG chain on first request."""
#     global embeddings, retriever, rag_chain

#     if rag_chain is not None:
#         return rag_chain  # already initialized

#     print("⏳ Initializing LangChain components (first request)...")

#     # Load embeddings model only once
#     embeddings = downloade_embeddings()

#     # Load existing Pinecone index (do NOT recreate)
#     index_name = "medical-chatbot"
#     docsearch = PineconeVectorStore.from_existing_index(
#         index_name=index_name,
#         embedding=embeddings
#     )

#     # Create retriever
#     retriever = docsearch.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 3}
#     )

#     # Initialize chat model (Groq)
#     chat_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

#     # Build prompt & chains
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}")
#     ])
#     question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     print("✅ RAG chain initialized successfully.")
#     return rag_chain

# # ---------------------------------------------------------
# # Routes
# # ---------------------------------------------------------
# @app.route("/")
# def index():
#     return render_template("chat.html")

# @app.route("/get", methods=["POST"])
# def chat():
#     global rag_chain
#     msg = request.form["msg"]

#     if rag_chain is None:
#         rag_chain = init_rag_chain()

#     print("User:", msg)
#     response = rag_chain.invoke({"input": msg})
#     answer = response.get("answer", "I'm sorry, I couldn't process that.")
#     print("Response:", answer)
#     return str(answer)

# # ---------------------------------------------------------
# # Run the Flask app
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)






import langchain_core
setattr(langchain_core, "memory", None)  # temporary patch for old modules

from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.helper import downloade_embeddings
from src.prompt import *
import os
import psutil


# ---------------------------------------------------------
# Utility: Memory usage checker
# ---------------------------------------------------------
def get_memory_usage():
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    mem_mb = mem_bytes / (1024 * 1024)
    return round(mem_mb, 2)


# ---------------------------------------------------------
# Flask setup
# ---------------------------------------------------------
app = Flask(__name__)
load_dotenv()


# ---------------------------------------------------------
# Environment variables
# ---------------------------------------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ---------------------------------------------------------
# Global variables (lazy-loaded objects)
# ---------------------------------------------------------
embeddings = None
retriever = None
rag_chain = None


# ---------------------------------------------------------
# Lazy initialization (only runs once, first request)
# ---------------------------------------------------------
def init_rag_chain():
    """Initialize embeddings, retriever, and RAG chain on first request."""
    global embeddings, retriever, rag_chain

    print("📦 Memory before loading:", get_memory_usage(), "MB")

    if rag_chain is not None:
        print("✅ RAG chain already loaded.")
        return rag_chain

    print("⏳ Initializing LangChain components (first request)...")

    # --- Load Hugging Face embeddings model ---
    embeddings = downloade_embeddings()
    print("📦 Memory after embeddings:", get_memory_usage(), "MB")

    # --- Load existing Pinecone index ---
    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print("📦 Memory after Pinecone load:", get_memory_usage(), "MB")

    # --- Create retriever ---
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # --- Initialize Groq model ---
    chat_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    print("📦 Memory after Groq init:", get_memory_usage(), "MB")

    # --- Build LangChain pipeline ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("🚀 Memory after full RAG chain init:", get_memory_usage(), "MB")
    print("✅ RAG chain initialized successfully.")
    return rag_chain


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    global rag_chain
    msg = request.form["msg"]

    if rag_chain is None:
        rag_chain = init_rag_chain()

    print("User:", msg)
    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer", "I'm sorry, I couldn't process that.")
    print("Response:", answer)
    print("📊 Current memory usage:", get_memory_usage(), "MB")

    return str(answer)


# ---------------------------------------------------------
# Run the Flask app
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
