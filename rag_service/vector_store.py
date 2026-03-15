"""Vector store (ChromaDB) and embeddings (sentence-transformers) for RAG. Heavy imports are lazy so the server starts fast."""
import logging
from pathlib import Path
from typing import List

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)

# Lazy-loaded singletons (avoid loading at server startup)
_embedding_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
    """Load sentence-transformers model once (lazy; only when embedding is needed)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_chroma_client():
    """Lazy-load ChromaDB client so server starts without waiting."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "AlphaMind RAG: model predictions and stock data"},
        )
    return _collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def add_documents(doc_ids: List[str], documents: List[str], metadatas: List[dict] = None):
    """Add documents to the vector store. Each doc is one chunk."""
    if metadatas is None:
        metadatas = [{}] * len(documents)
    if len(metadatas) != len(documents):
        metadatas = [{}] * len(documents)
    coll = get_collection()
    embeddings = embed_texts(documents)
    coll.upsert(
        ids=doc_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(f"Upserted {len(documents)} documents into ChromaDB.")


def query_documents(query: str, top_k: int = 5) -> List[dict]:
    """Return top_k most relevant chunks for the query."""
    coll = get_collection()
    q_embedding = embed_texts([query])[0]
    results = coll.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, 20),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            out.append({
                "content": doc,
                "metadata": (results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}),
                "distance": results["distances"][0][i] if results.get("distances") and results["distances"][0] else None,
            })
    return out


def collection_count() -> int:
    return get_collection().count()
