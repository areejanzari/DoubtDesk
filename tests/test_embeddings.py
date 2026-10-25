import numpy as np
import pytest
from unittest.mock import MagicMock
from doubtdesk.ingestion.embeddings import embed_chunks
from doubtdesk.retrieval.search import embed_query


def test_embed_chunks():
    model = MagicMock()
    model.encode.return_value = np.array([
        [0.6, 0.8],
        [0.8, 0.6],
    ])

    chunks = [
        MagicMock(text="Photosynthesis"),
        MagicMock(text="Cell division"),
    ]

    result = embed_chunks(chunks, model)

    assert result.shape == (2, 2)
    model.encode.assert_called_once_with(
        ["Photosynthesis", "Cell division"],
        normalize_embeddings=True,
    )


def test_embed_empty_chunks():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384

    result = embed_chunks([], model)

    assert result.shape == (0, 384)
    model.encode.assert_not_called()


def test_embed_query():
    model = MagicMock()
    model.encode.return_value = np.array([0.6, 0.8])

    result = embed_query("What is photosynthesis?", model)

    assert result.shape == (2,)
    model.encode.assert_called_once_with(
        "What is photosynthesis?",
        normalize_embeddings=True,
    )

@pytest.mark.parametrize("question", ["", " ", "\n\t"])
def test_empty_query(question):
    model = MagicMock()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        embed_query(question, model)

    model.encode.assert_not_called()