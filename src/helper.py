# from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs( docs:List[Document]) -> List[Document]:
    """
    given a list of documents that are filtered based on sources
    """

    minmal_docs:List[Document]=[]
    for doc in docs:
        src = doc.metadata.get("source")
        minmal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )

    return minmal_docs


# splitting the documents into smaller chunks 
def split_documents(minmal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(minmal_docs)
    return text_chunks


def downloade_embeddings():
    """
    download the embedding model from the hugging face
    """

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name,
    )

    return embeddings

embeddings = downloade_embeddings()