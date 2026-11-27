from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from doubtdesk.db import EMBEDDING_DIMENSIONS
from doubtdesk.graph.nodes import retrieve_node
from doubtdesk.ingestion.schemas import ChunkRecord
from doubtdesk.ingestion.registry import (
    attach_document_to_user,
    register_document,
    update_document_status,
)
from doubtdesk.retrieval.search import search_chunks
from doubtdesk.retrieval.vector_store import replace_document_chunks


def unit(index):
    vector = np.zeros(EMBEDDING_DIMENSIONS)
    vector[index] = 1.0
    return vector


def test_retrieve_node_rejects_other_user_before_embedding():
    register_document("doc_A", "biology.pdf", "hash_A")
    update_document_status("doc_A", "completed", 3)
    attach_document_to_user("user_A", "doc_A", "biology.pdf")

    model = MagicMock()
    retrieve = retrieve_node(model)

    with patch("doubtdesk.graph.nodes.aretrieve_ranked_evidence") as search:
        with pytest.raises(ValueError, match="unavailable"):
            retrieve({"user_id": "user_B", "question": "Summarize", "doc_ids": ["doc_A"]})

    model.encode.assert_not_called()
    search.assert_not_called()

    assert retrieve({
        "user_id": "user_B", "question": "General question", "doc_ids": []
    }) == {"retrieved_chunks": [], "retrieved_pages": []}


def test_document_filter_overrides_vector_similarity():
    replace_document_chunks(
        "doc_A",
        [ChunkRecord(doc_id="doc_A", page_number=1, chunk_index=0, text="Student A content")],
        np.array([unit(0)]),
    )
    distractors = [
        ChunkRecord(doc_id="doc_B", page_number=page, chunk_index=0, text="Student B content")
        for page in range(1, 201)
    ]
    replace_document_chunks("doc_B", distractors, np.array([unit(1)] * len(distractors)))

    result = search_chunks(unit(1), doc_ids=["doc_A"], top_k=5)

    assert [item["chunk_id"] for item in result[0]] == ["doc_A_p1_c0"]
