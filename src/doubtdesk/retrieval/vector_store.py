from pathlib import Path
import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from doubtdesk.ingestion.schemas import ChunkRecord

def get_collection(
        store_path: Path = Path("data/chroma")
) -> Collection:
    client = chromadb.PersistentClient(path=str(store_path))

    collection = client.get_or_create_collection(
        name="doubtdesk_chunks",
        configuration={"hnsw": {"space": "cosine"}},
    )

    return collection

def store_chunks(
    collection: Collection,
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
) -> None:
    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count must match")

    ids = [
        f"{chunk.doc_id}_p{chunk.page_number}_c{chunk.chunk_index}"
        for chunk in chunks
    ]

    documents = [chunk.text for chunk in chunks]

    metadatas = [
        {
            "doc_id": chunk.doc_id,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )


def delete_document_chunks(
    collection: Collection,
    doc_id: str,
) -> None:
    collection.delete(
        where={"doc_id": doc_id}
    )

