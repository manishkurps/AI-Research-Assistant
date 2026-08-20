import asyncio
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from retrieval_graph.graph import graph
from shared.ingestion import ingest_uploaded_pdfs


MAX_FILES = 10
MAX_FILE_SIZE_MB = 20


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)


st.title("🔬 AI Research Assistant")
st.caption(
    "Upload research papers and ask questions using a LangGraph-powered RAG agent."
)

# --------------------------------------------------
# Response formatting
# --------------------------------------------------

def extract_response_text(content):
    """Extract clean text from the model response."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))

        return "\n".join(texts)

    return str(content)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False


# --------------------------------------------------
# Sidebar - PDF upload
# --------------------------------------------------

with st.sidebar:
    st.header("📚 Research Papers")

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum {MAX_FILES} PDFs, {MAX_FILE_SIZE_MB} MB each.",
    )

    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"You can upload a maximum of {MAX_FILES} PDFs.")
        else:
            oversized_files = [
                file.name
                for file in uploaded_files
                if file.size > MAX_FILE_SIZE_MB * 1024 * 1024
            ]

            if oversized_files:
                st.error(
                    "These files exceed the 20 MB limit:\n\n"
                    + "\n".join(oversized_files)
                )
            else:
                if st.button(
                    "📥 Process Research Papers",
                    use_container_width=True,
                ):
                    with st.spinner("Processing research papers..."):
                        temp_paths = []

                        try:
                            for uploaded_file in uploaded_files:
                                temp_file = tempfile.NamedTemporaryFile(
                                    delete=False,
                                    suffix=".pdf",
                                )

                                temp_file.write(uploaded_file.getvalue())
                                temp_file.close()

                                temp_paths.append(
                                    Path(temp_file.name)
                                )

                            ingest_uploaded_pdfs(
                                temp_paths,
                                [file.name for file in uploaded_files],
                            )

                            st.session_state.documents_ready = True
                            st.session_state.messages = []

                            st.success(
                                f"Successfully processed "
                                f"{len(uploaded_files)} paper(s)."
                            )

                        except Exception as exc:
                            st.error(
                                f"Failed to process documents: {exc}"
                            )

                        finally:
                            for path in temp_paths:
                                try:
                                    path.unlink(missing_ok=True)
                                except Exception:
                                    pass

    if st.session_state.documents_ready:
        st.success("Research papers are ready.")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Display previous conversation
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(source)


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your research papers..."
)


if question:

    if not st.session_state.documents_ready:
        st.warning(
            "Please upload and process at least one research paper first."
        )
        st.stop()

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("assistant"):

        with st.spinner("Researching your question..."):

            try:
                result = asyncio.run(
                    graph.ainvoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content=message["content"]
                                )
                                for message in st.session_state.messages
                                if message["role"] == "user"
                            ]
                        }
                    )
                )

                response = result["messages"][-1]
                answer = extract_response_text(response.content)

                st.markdown(answer)

                # Collect source information
                sources = []

                for document in result.get("documents", []):
                    source = document.metadata.get(
                        "source",
                        "Unknown source",
                    )

                    page = document.metadata.get(
                        "page",
                        "Unknown page",
                    )

                    source_text = f"{source} — Page {page}"

                    if source_text not in sources:
                        sources.append(source_text)

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"- {source}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                st.error(
                    f"Something went wrong while answering the question: {exc}"
                )