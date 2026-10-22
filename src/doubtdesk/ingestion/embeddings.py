import numpy as np
from sentence_transformers import SentenceTransformer
from doubtdesk.ingestion.schemas import ChunkRecord

def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(
        chunks: list[ChunkRecord],
        model: SentenceTransformer,
) -> np.ndarray:
    if not chunks:
        return np.empty((0, model.get_sentence_embedding_dimension()))

    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    return embeddings