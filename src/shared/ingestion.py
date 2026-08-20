"""Document ingestion pipeline for the AI Research Assistant."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ai_research_documents"

DOCUMENTS_DIR = Path("./data/documents")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

def create_embeddings():
    """Create the local HuggingFace embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# --------------------------------------------------
# PDF loading
# --------------------------------------------------

def load_pdf(pdf_path: Path) -> list[Document]:
    """Load text from a PDF file."""

    reader = PdfReader(str(pdf_path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    return documents


# --------------------------------------------------
# Load all PDFs
# --------------------------------------------------

def load_documents() -> list[Document]:
    """Load all PDF documents from the documents directory."""

    documents = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR.resolve()}"
        )

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        documents.extend(load_pdf(pdf_path))

    return documents


# --------------------------------------------------
# Split documents into chunks
# --------------------------------------------------

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# Store documents in ChromaDB
# --------------------------------------------------

def create_vector_store(documents: list[Document]):
    """Create a fresh ChromaDB vector store from document chunks."""

    embeddings = create_embeddings()

    # Create a fresh collection every time ingestion runs.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Remove existing documents to prevent duplicates.
    try:
        existing = vector_store.get()

        if existing and existing.get("ids"):
            vector_store.delete(ids=existing["ids"])
            print(f"Removed {len(existing['ids'])} existing chunks.")

    except Exception as e:
        print(f"Could not clear existing collection: {e}")

    # Add the newly processed chunks.
    vector_store.add_documents(documents)

    return vector_store


# --------------------------------------------------
# Main ingestion pipeline
# --------------------------------------------------

def clear_vector_store():
    """Clear all documents from the research document collection."""

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vector_store.delete_collection()

    print("Existing research collection cleared.")



def ingest_uploaded_pdfs(
    pdf_paths: list[Path],
    original_names: list[str],
):
    """Ingest uploaded PDF files into the ChromaDB collection."""

    print("\n=== UPLOADED DOCUMENT INGESTION ===")

    clear_vector_store()

    documents = []

    for pdf_path, original_name in zip(pdf_paths, original_names):
        print(f"Loading: {original_name}")

        loaded_documents = load_pdf(pdf_path)

        for document in loaded_documents:
            document.metadata["source"] = original_name

        documents.extend(loaded_documents)

    print(f"Loaded {len(documents)} pages.")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    vector_store = create_vector_store(chunks)

    print("Uploaded documents stored successfully.")

    return vector_store


def ingest_documents():
    """Run the complete document ingestion pipeline."""

    print("\n=== DOCUMENT INGESTION ===")

    print("\n1. Loading PDFs...")
    documents = load_documents()

    print(f"Loaded {len(documents)} pages.")

    print("\n2. Splitting documents...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("\n3. Creating embeddings and storing in ChromaDB...")
    vector_store = create_vector_store(chunks)

    print("Documents stored successfully.")

    print("\n4. Checking collection...")
    print(f"Collection: {COLLECTION_NAME}")

    try:
        print(f"Documents in collection: {vector_store._collection.count()}")
    except Exception:
        pass

    print("\n=== INGESTION COMPLETE ===")


if __name__ == "__main__":
    ingest_documents()