import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from doubtdesk.retrieval.vector_store import (
    get_collection,
    store_chunks,
    delete_document_chunks,
)
from doubtdesk.retrieval.search import (
    search_chunks,
    format_search_results,
    embed_queries
)
from doubtdesk.retrieval.pages import (
    get_page_image_paths,
)


@pytest.fixture
def collection(tmp_path):
    return get_collection(tmp_path / "chroma")


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
def populated_collection(collection, sample_chunks):
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
    ])

    store_chunks(collection, sample_chunks, embeddings)
    return collection

def test_store_chunks(populated_collection):
    result = populated_collection.get()

    assert len(result["ids"]) == 3

    assert set(result["ids"]) == {
        "doc_001_p1_c0",
        "doc_001_p2_c0",
        "doc_002_p1_c0",
    }


def test_delete_document_chunks(populated_collection):
    delete_document_chunks(populated_collection, "doc_001")

    result = populated_collection.get()

    assert result["ids"] == ["doc_002_p1_c0"]


def test_mismatched_embeddings(collection, sample_chunks):
    embeddings = np.array([[1.0, 0.0, 0.0]])

    with pytest.raises(
        ValueError,
        match="Chunks and embeddings count must match",
    ):
        store_chunks(collection, sample_chunks, embeddings)


def test_store_empty_chunks(collection):
    store_chunks(collection, [], np.empty((0, 3)))

    assert collection.count() == 0

def test_search_chunks(populated_collection):
    query = np.array([1.0, 0.0, 0.0])

    results = search_chunks(
        populated_collection,
        query,
        doc_ids=["doc_001"],
        top_k=2,
    )

    formatted = format_search_results(results)

    assert len(formatted) == 2
    assert all(
        result["doc_id"] == "doc_001"
        for result in formatted
    )

    assert formatted[0]["text"] == "Photosynthesis"
    assert formatted[0]["page_number"] == 1


def test_search_multiple_documents(populated_collection):
    query = np.array([0.0, 1.0, 0.0])

    results = search_chunks(
        populated_collection,
        query,
        doc_ids=["doc_001", "doc_002"],
        top_k=1,
    )

    formatted = format_search_results(results)

    assert formatted[0]["doc_id"] == "doc_002"
    assert formatted[0]["text"] == "Newton's laws"

@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k(populated_collection, top_k):
    with pytest.raises(ValueError, match="top_k must be positive"):
        search_chunks(
            populated_collection,
            np.array([1.0, 0.0, 0.0]),
            ["doc_001"],
            top_k=top_k,
        )


def test_empty_document_ids(populated_collection):
    with pytest.raises(
        ValueError,
        match="Atleast one document ID is required",
    ):
        search_chunks(
            populated_collection,
            np.array([1.0, 0.0, 0.0]),
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

def test_batched_search_matches_individual_queries(populated_collection):
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
    ])

    batched = search_chunks(
        populated_collection,
        embeddings,
        doc_ids=["doc_001", "doc_002"],
        top_k=2,
    )

    assert len(batched["ids"]) == 3

    for index, embedding in enumerate(embeddings):
        individual = search_chunks(
            populated_collection,
            embedding,
            doc_ids=["doc_001", "doc_002"],
            top_k=2,
        )

        assert batched["ids"][index] == individual["ids"][0]

    formatted = format_search_results(batched)

    assert len(formatted) == 6