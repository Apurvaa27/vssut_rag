"""Command-line entry point for local document RAG."""

import argparse
import sys

from src.config import settings
from src.ingestion import load_documents, split_documents
from src.rag import RAG
from src.vector_store import VectorStore


def make_vector_store() -> VectorStore:
    return VectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
    )


def ingest() -> None:
    documents = load_documents(settings.data_dir)
    if not documents:
        print(f"No PDF, TXT, or MD files found in {settings.data_dir}")
        print("Copy documents into the data/ folder, then run ingestion again.")
        return
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    store = make_vector_store()
    store.clear()
    added = store.add_documents(chunks)
    print(f"Loaded {len(documents)} pages/documents and indexed {added} chunks.")
    sources = sorted({doc.metadata.get("source", "unknown") for doc in documents})
    print("Indexed sources:")
    for source in sources:
        print(f"- {source}")
    print(f"Vector database: {settings.chroma_dir}")


def ask(question: str) -> None:
    if not settings.google_api_key:
        sys.exit("GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key.")
    rag = RAG(make_vector_store(), settings.google_api_key, settings.gemini_model)
    answer, sources = rag.ask(question, settings.top_k)
    print(f"\n{answer}\n")
    if sources:
        print("Sources:")
        seen = set()
        citation_number = 0
        for match in sources:
            source = match["metadata"].get("source", "unknown")
            page = match["metadata"].get("page")
            citation = (source, page)
            if citation in seen:
                continue
            seen.add(citation)
            citation_number += 1
            page_text = f", page {page + 1}" if isinstance(page, int) else ""
            print(f"[{citation_number}] {source}{page_text}")


def chat() -> None:
    if not settings.google_api_key:
        sys.exit("GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key.")
    rag = RAG(make_vector_store(), settings.google_api_key, settings.gemini_model)
    print("Ask questions about your indexed documents. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer, sources = rag.ask(question, settings.top_k)
            print(f"\nAssistant: {answer}")
            seen = set()
            citation_number = 0
            for match in sources:
                source = match["metadata"].get("source", "unknown")
                page = match["metadata"].get("page")
                citation = (source, page)
                if citation in seen:
                    continue
                seen.add(citation)
                citation_number += 1
                page_text = f", page {page + 1}" if isinstance(page, int) else ""
                print(f"  [{citation_number}] {source}{page_text}")


def retrieve(question: str) -> None:
    """Show the passages selected by vector search without calling Gemini."""
    store = make_vector_store()
    matches = store.search(question, settings.top_k)
    if not matches:
        print("No indexed passages found. Run: python app.py ingest")
        return
    for match in matches:
        metadata = match["metadata"]
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        page_text = f", page {page + 1}" if isinstance(page, int) else ""
        print(
            f"\n[{match['rank']}] {source}{page_text} "
            f"(similarity {match['similarity_score']:.3f})"
        )
        print(match["document"][:1200])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your documents")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="Index files in the data/ folder")
    ask_parser = commands.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question", nargs="+", help="Question text")
    retrieve_parser = commands.add_parser(
        "retrieve", help="Inspect passages retrieved for a question (no Gemini call)"
    )
    retrieve_parser.add_argument("question", nargs="+", help="Question text")
    commands.add_parser("chat", help="Start an interactive Q&A session")
    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "ask":
        ask(" ".join(args.question))
    elif args.command == "retrieve":
        retrieve(" ".join(args.question))
    else:
        chat()


if __name__ == "__main__":
    main()
