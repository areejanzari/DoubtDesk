from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from doubtdesk.graph.nodes import retrieve_node
from doubtdesk.ingestion.registry import (
    attach_document_to_user,
    initialize_registry,
    register_document,
    update_document_status,
)
from doubtdesk.retrieval.search import search_chunks


def test_retrieve_node_rejects_other_user_before_embedding(tmp_path):
    db_path = tmp_path / "documents.db"
    initialize_registry(db_path)
    register_document("doc_A", "biology.pdf", "hash_A", db_path)
    update_document_status("doc_A", "completed", 3, db_path)
    attach_document_to_user("user_A", "doc_A", "biology.pdf", db_path)

    model = MagicMock()
    collection = MagicMock()
    retrieve = retrieve_node(collection, model, db_path)

    with pytest.raises(ValueError, match="unavailable"):
        retrieve({"user_id": "user_B", "question": "Summarize", "doc_ids": ["doc_A"]})

    model.encode.assert_not_called()
    collection.query.assert_not_called()

    assert retrieve({
        "user_id": "user_B", "question": "General question", "doc_ids": []
    }) == {"retrieved_pages": []}


def test_chroma_document_filter_overrides_vector_similarity():
    import chromadb

    collection = chromadb.Client().get_or_create_collection(
        name=f"authorization_{uuid4().hex[:12]}"
    )
    collection.add(
        ids=["chunk_A", "chunk_B"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["Student A content", "Student B content"],
        metadatas=[{"doc_id": "doc_A"}, {"doc_id": "doc_B"}],
    )

    result = search_chunks(
        collection,
        np.array([0.0, 1.0]),
        doc_ids=["doc_A"],
        top_k=5,
    )

    assert result["ids"] == [["chunk_A"]]
