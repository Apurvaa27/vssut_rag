"""Embedding generation and persistent Chroma operations."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str) -> SentenceTransformer:
    """Share the large embedding model across stores in this process."""
    return SentenceTransformer(model_name)


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedding_model: str):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.model = _load_embedding_model(embedding_model)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks for local RAG"},
            # Chroma uses cosine distance; retrieval score is 1 - distance.
            configuration={"hnsw": {"space": "cosine"}},
        )

    def clear(self) -> None:
        """Remove the current index before a full re-ingestion."""
        count = self.collection.count()
        ids = self.collection.get(limit=count, include=[])["ids"] if count else []
        for start in range(0, len(ids), 1000):
            self.collection.delete(ids=ids[start : start + 1000])

    def add_documents(self, documents: list[Document], batch_size: int = 128) -> int:
        if not documents:
            return 0
        texts = [doc.page_content for doc in documents]
        vectors = self.model.encode(texts, show_progress_bar=True).tolist()

        # Stable IDs let re-ingestion update unchanged chunks instead of
        # accumulating duplicate copies in the persistent collection.
        metadata = []
        ids = []
        for index, doc in enumerate(documents):
            item = {key: value for key, value in doc.metadata.items() if value is not None}
            item["chunk_index"] = index
            metadata.append(item)
            identity = "|".join(
                [str(item.get("source", "")), str(item.get("page", "")), str(index), doc.page_content]
            )
            ids.append(f"chunk_{sha256(identity.encode('utf-8')).hexdigest()}")

        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=vectors[start:end],
                metadatas=metadata[start:end],
            )
        return len(documents)

    def search(self, query: str, top_k: int) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        query_vector = self.model.encode([query])[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        matches = []
        for rank, (text, metadata, distance) in enumerate(
            zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            ),
            start=1,
        ):
            matches.append(
                {
                    "document": text,
                    "metadata": metadata or {},
                    "distance": distance,
                    "similarity_score": 1.0 - distance,
                    "rank": rank,
                }
            )
        return matches