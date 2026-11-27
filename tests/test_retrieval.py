import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from doubtdesk.db import EMBEDDING_DIMENSIONS, connection
from doubtdesk.retrieval.vector_store import (
    replace_document_chunks,
    document_has_chunks,
)
from doubtdesk.retrieval.search import (
    search_chunks,
    embed_queries
)
from doubtdesk.retrieval.pages import (
    get_page_image_paths,
)


def vector(*values):
    padded = np.zeros(EMBEDDING_DIMENSIONS)
    padded[:len(values)] = values
    return padded


def stored_chunk_ids():
    with connection() as conn:
        return {row[0] for row in conn.execute("SELECT chunk_id FROM chunks")}


@pytest.fixture
def sample_chunks():
    chunks = []

    for doc_id, page_number, chunk_index, text in [
        ("doc_001", 1, 0, "Photosynthesis"),
        ("doc_001", 2, 0, "Chlorophyll"),
        ("doc_002", 1, 0, "Newton's laws"),
    ]:
        chunk = MagicMock()
        chunk.doc_id = doc_id
        chunk.page_number = page_number
        chunk.chunk_index = chunk_index
        chunk.text = text
        chunks.append(chunk)

    return chunks


@pytest.fixture
def populated(sample_chunks):
    replace_document_chunks("doc_001", sample_chunks[:2], np.array([vector(1.0, 0.0), vector(0.9, 0.1)]))
    replace_document_chunks("doc_002", sample_chunks[2:], np.array([vector(0.0, 1.0)]))


def test_store_chunks(populated):
    assert stored_chunk_ids() == {
        "doc_001_p1_c0",
        "doc_001_p2_c0",
        "doc_002_p1_c0",
    }
    assert document_has_chunks("doc_001")
    assert not document_has_chunks("doc_missing")


def test_replacing_document_removes_previous_chunks(populated, sample_chunks):
    replace_document_chunks("doc_001", [], np.empty((0, EMBEDDING_DIMENSIONS)))

    assert stored_chunk_ids() == {"doc_002_p1_c0"}
    assert not document_has_chunks("doc_001")


def test_mismatched_embeddings(sample_chunks):
    embeddings = np.array([vector(1.0)])

    with pytest.raises(
        ValueError,
        match="Chunks and embeddings count must match",
    ):
        replace_document_chunks("doc_001", sample_chunks, embeddings)


def test_search_chunks(populated):
    results = search_chunks(
        vector(1.0, 0.0),
        doc_ids=["doc_001"],
        top_k=2,
    )

    assert len(results) == 1
    formatted = results[0]
    assert len(formatted) == 2
    assert all(
        result["doc_id"] == "doc_001"
        for result in formatted
    )

    assert formatted[0]["text"] == "Photosynthesis"
    assert formatted[0]["page_number"] == 1
    assert formatted[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert formatted[0]["distance"] < formatted[1]["distance"]


def test_search_multiple_documents(populated):
    formatted = search_chunks(
        vector(0.0, 1.0),
        doc_ids=["doc_001", "doc_002"],
        top_k=1,
    )[0]

    assert formatted[0]["doc_id"] == "doc_002"
    assert formatted[0]["text"] == "Newton's laws"

@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k(populated, top_k):
    with pytest.raises(ValueError, match="top_k must be positive"):
        search_chunks(
            vector(1.0),
            ["doc_001"],
            top_k=top_k,
        )


def test_empty_document_ids(populated):
    with pytest.raises(
        ValueError,
        match="Atleast one document ID is required",
    ):
        search_chunks(
            vector(1.0),
            [],
        )



def test_get_page_image_paths(tmp_path):
    doc_dir = tmp_path / "doc1"
    doc_dir.mkdir()

    image = doc_dir / "page_003.png"
    image.write_bytes(b"fake-image-data")

    pages = [
        {"doc_id": "doc1", "page_number": 3},
    ]

    result = get_page_image_paths(
        pages,
        image_root=tmp_path,
    )

    assert len(result) == 1
    assert result[0]["doc_id"] == "doc1"
    assert result[0]["page_number"] == 3
    assert result[0]["image_path"] == image
    assert isinstance(result[0]["image_path"], Path)


def test_get_page_image_paths_missing_file(tmp_path):
    pages = [
        {"doc_id": "doc1", "page_number": 3},
    ]

    with pytest.raises(FileNotFoundError, match="Page image not found"):
        get_page_image_paths(
            pages,
            image_root=tmp_path,
        )


def test_get_page_image_paths_empty(tmp_path):
    assert get_page_image_paths([], image_root=tmp_path) == []

def test_embed_queries_uses_single_batch():
    model = MagicMock()
    model.encode.return_value = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])

    questions = ["Question 1", "Question 2", "Question 3"]

    embeddings = embed_queries(questions, model)

    model.encode.assert_called_once_with(
        questions,
        normalize_embeddings=True,
    )

    assert embeddings.shape == (3, 2)

def test_batched_search_matches_individual_queries(populated):
    embeddings = np.array([
        vector(1.0, 0.0),
        vector(0.9, 0.1),
        vector(0.0, 1.0),
    ])

    batched = search_chunks(
        embeddings,
        doc_ids=["doc_001", "doc_002"],
        top_k=2,
    )

    assert len(batched) == 3

    for index, embedding in enumerate(embeddings):
        individual = search_chunks(
            embedding,
            doc_ids=["doc_001", "doc_002"],
            top_k=2,
        )

        assert batched[index] == individual[0]

    assert sum(len(results) for results in batched) == 6
