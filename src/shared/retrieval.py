"""Local ChromaDB retriever using HuggingFace embeddings."""

from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ai_research_documents"

# Free local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Shared embedding model
# --------------------------------------------------

_embeddings: Embeddings | None = None


def make_text_encoder() -> Embeddings:
    """Create the local HuggingFace embedding model once."""

    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


# --------------------------------------------------
# Retriever
# --------------------------------------------------

@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a local ChromaDB retriever."""

    embeddings = make_text_encoder()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}
    )

    yield retriever