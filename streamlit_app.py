"""Chat-only Streamlit interface over the project owner's document collection."""

import hashlib
import os
import time
from pathlib import Path

import streamlit as st

from src.config import settings
from src.ingestion import load_documents, split_documents
from src.rag import RAG
from src.vector_store import VectorStore


st.set_page_config(page_title="VSSUT RAG", page_icon="📚", layout="wide")


def get_api_key() -> str | None:
    if settings.google_api_key:
        return settings.google_api_key
    try:
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        # Local runs may use only .env and have no Streamlit secrets file.
        return os.getenv("GOOGLE_API_KEY")


def get_dataset_signature() -> str:
    """Make Streamlit rebuild the index when the background files change."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in sorted(settings.data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md", ".php"}:
            # A same-name Markdown file is generated from this PHP source. The
            # PHP file already fingerprints its contents, so ignore the output
            # here to prevent a redundant reindex immediately after conversion.
            if path.suffix.lower() == ".md" and path.with_suffix(".php").is_file():
                continue
            stat = path.stat()
            relative_path = path.relative_to(settings.data_dir).as_posix()
            digest.update(f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


@st.cache_resource(max_entries=1, show_spinner="Wait while I'm loading")
def prepare_knowledge_base(dataset_signature: str) -> tuple[VectorStore, int, int]:
    """Index owner-provided files once per process and whenever they change."""
    store = VectorStore(
        settings.chroma_dir,
        settings.collection_name,
        settings.embedding_model,
    )
    store.clear()
    documents = load_documents(settings.data_dir)
    if not documents:
        return store, 0, 0
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    store.add_documents(chunks)
    return store, len(documents), len(chunks)


@st.cache_resource(show_spinner=False)
def get_rag(dataset_signature: str, api_key: str) -> RAG:
    store, _, _ = prepare_knowledge_base(dataset_signature)
    return RAG(store, api_key, settings.gemini_model)


st.title("📚 VSSUT Assistant")
st.caption("Ask anything you want to know about VSSUT")

dataset_signature = get_dataset_signature()
try:
    vector_store, document_count, chunk_count = prepare_knowledge_base(dataset_signature)
except Exception as exc:
    st.error(f"Could not prepare the document knowledge base: {exc}")
    st.stop()

api_key = get_api_key()
if not document_count or not chunk_count:
    st.info("The knowledge base is empty. The project owner needs to add PDF, TXT, Markdown, or PHP files to `data/` and redeploy.")
# elif api_key:
#     st.caption(f"Knowledge base ready · {document_count} pages/documents · {chunk_count} searchable passages")
# else:
#     st.warning("The app owner needs to configure GOOGLE_API_KEY in `.env` or Streamlit Cloud secrets.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("About this assistant")
    st.write("Answers are generated from the project owner's indexed documents.")
    if document_count:
        st.caption(f"{document_count} pages/documents indexed")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # if message.get("sources"):
        #     with st.expander("Sources"):
        #         for source in message["sources"]:
        #             page = source.get("page")
        #             page_label = f", page {page + 1}" if isinstance(page, int) else ""
        #             st.markdown(f"- **{source['name']}**{page_label}")

chat_enabled = bool(document_count and chunk_count and api_key)
question = st.chat_input("Ask a question...", disabled=not chat_enabled)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            started_at = time.perf_counter()
            with st.spinner("Wait while I search for your answer..."):
                rag = get_rag(dataset_signature, api_key)
                answer, matches = rag.ask(question, settings.top_k)
            st.markdown(answer)
            elapsed = time.perf_counter() - started_at
            # st.caption(f"Retrieved {len(matches)} passages · {elapsed:.1f}s")
            # sources = []
            # seen = set()
            # for match in matches:
            #     metadata = match["metadata"]
            #     source = metadata.get("source", "unknown")
            #     page = metadata.get("page")
            #     citation = (source, page)
            #     if citation not in seen:
            #         seen.add(citation)
            #         sources.append({"name": source, "page": page})
            # if sources:
            #     with st.expander("Sources"):
            #         for source in sources:
            #             page = source["page"]
            #             page_label = f", page {page + 1}" if isinstance(page, int) else ""
            #             st.markdown(f"- **{source['name']}**{page_label}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as exc:
            st.error(f"The question could not be answered: {exc}")