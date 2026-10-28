import hashlib
from pathlib import Path
import pymupdf
import pytest
from unittest.mock import patch, MagicMock
from doubtdesk.ingestion.hashing import calculate_sha256
from doubtdesk.ingestion.pdf_reader import render_pdf_pages
from doubtdesk.ingestion.pipeline import ingest_document
from doubtdesk.ingestion.service import process_pdf
from doubtdesk.ingestion.cache import cached_artifacts_exist


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        for heading in ["Photosynthesis", "Cell division"]:
            page = pdf.new_page()
            for i in range(25):
                page.insert_text(
                    (50, 50 + i * 20),
                    f"{heading} is an important biological process.",
                )
        pdf.save(pdf_path)

    return pdf_path

@pytest.fixture
def service_mocks():
    with (
        patch("doubtdesk.ingestion.service.initialize_registry"),
        patch(
            "doubtdesk.ingestion.service.calculate_sha256",
            return_value="hash_001",
        ),
        patch(
            "doubtdesk.ingestion.service.get_collection",
            return_value=MagicMock(),
        ),
        patch(
            "doubtdesk.ingestion.service.get_document_by_hash",
            return_value=None,
        ) as get_document,
        patch(
            "doubtdesk.ingestion.service.register_document"
        ) as register,
        patch(
            "doubtdesk.ingestion.service.update_document_status"
        ) as update_status,
        patch(
            "doubtdesk.ingestion.service.ingest_document",
            return_value=["page1", "page2"],
        ) as ingest,
        patch(
            "doubtdesk.ingestion.service.chunk_document",
            return_value=["chunk1"],
        ),
        patch("doubtdesk.ingestion.service.load_embedding_model"),
        patch(
            "doubtdesk.ingestion.service.embed_chunks",
            return_value=[[0.1, 0.2]],
        ),
        patch(
            "doubtdesk.ingestion.service.delete_document_chunks"
        ) as delete_chunks,
        patch(
            "doubtdesk.ingestion.service.store_chunks"
        ) as store,
    ):
        yield {
            "get_document": get_document,
            "register": register,
            "update_status": update_status,
            "ingest": ingest,
            "delete_chunks": delete_chunks,
            "store": store,
        }

def test_sha256_hash(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"Hello DoubtDesk")

    expected = hashlib.sha256(b"Hello DoubtDesk").hexdigest()

    assert calculate_sha256(file_path) == expected


def test_sha256_changes_with_content(tmp_path):
    file_path = tmp_path / "sample.txt"

    file_path.write_bytes(b"First version")
    first_hash = calculate_sha256(file_path)

    file_path.write_bytes(b"Second version")
    second_hash = calculate_sha256(file_path)

    assert first_hash != second_hash


def test_render_pdf_pages(sample_pdf, tmp_path):
    output_dir = tmp_path / "images"

    image_paths = render_pdf_pages(
        sample_pdf, output_dir, dpi=72
    )

    assert len(image_paths) == 2
    assert [path.name for path in image_paths] == [
        "page_001.png",
        "page_002.png",
    ]
    assert all(path.is_file() for path in image_paths)

    with pymupdf.open(image_paths[0]) as image:
        assert image.page_count == 1

def test_ingest_document(sample_pdf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    records = ingest_document(
        pdf_path=sample_pdf,
        doc_id="doc_001",
    )

    assert len(records) == 2

    assert [record.page_number for record in records] == [1, 2]
    assert all(record.doc_id == "doc_001" for record in records)

    assert "Photosynthesis" in records[0].text
    assert "Cell division" in records[1].text

    assert all(
        record.classification == "good_text"
        for record in records
    )

    assert all(
        Path(record.image_path).is_file()
        for record in records
    )

def test_successful_ingestion(service_mocks):
    result = process_pdf(Path("biology.pdf"))

    assert result["status"] == "completed"
    assert result["page_count"] == 2

    service_mocks["register"].assert_called_once()
    service_mocks["ingest"].assert_called_once()

    service_mocks["update_status"].assert_any_call(
        result["doc_id"], "processing", 0
    )
    service_mocks["update_status"].assert_any_call(
        result["doc_id"], "completed", 2
    )

    service_mocks["delete_chunks"].assert_called_once()
    service_mocks["store"].assert_called_once()


def test_cached_document(service_mocks):
    service_mocks["get_document"].return_value = {
        "doc_id": "doc_001",
        "page_count": 8,
    }

    with (
        patch(
            "doubtdesk.ingestion.service.is_document_cached",
            return_value=True,
        ),
        patch(
            "doubtdesk.ingestion.service.cached_artifacts_exist",
            return_value=True,
        ),
    ):
        result = process_pdf(Path("biology.pdf"))

    assert result == {
        "doc_id": "doc_001",
        "status": "cached",
        "page_count": 8,
    }

    service_mocks["ingest"].assert_not_called()
    service_mocks["register"].assert_not_called()
    service_mocks["store"].assert_not_called()

def test_ingestion_failure(service_mocks):
    service_mocks["ingest"].side_effect = RuntimeError(
        "PDF extraction failed"
    )

    with pytest.raises(RuntimeError, match="PDF extraction failed"):
        process_pdf(Path("biology.pdf"))

    doc_id = service_mocks["register"].call_args.kwargs["doc_id"]

    service_mocks["update_status"].assert_any_call(
        doc_id, "failed", 0
    )

    service_mocks["store"].assert_not_called()


def test_empty_chunks(service_mocks):
    with patch(
        "doubtdesk.ingestion.service.chunk_document",
        return_value=[],
    ):
        result = process_pdf(Path("biology.pdf"))

    assert result["status"] == "completed"
    service_mocks["delete_chunks"].assert_called_once()
    service_mocks["store"].assert_not_called()


@pytest.mark.parametrize("page_count", [0, -1])
def test_cache_invalid_page_count(tmp_path, monkeypatch, page_count):
    monkeypatch.chdir(tmp_path)
    collection = MagicMock()

    result = cached_artifacts_exist("doc1", page_count, collection)

    assert result is False
    collection.get.assert_not_called()


def test_cache_missing_image_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    collection = MagicMock()

    result = cached_artifacts_exist("doc1", 2, collection)

    assert result is False
    collection.get.assert_not_called()


def test_cache_missing_page_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    image_dir = tmp_path / "data/page_images/doc1"
    image_dir.mkdir(parents=True)
    (image_dir / "page_001.png").write_bytes(b"image")

    collection = MagicMock()

    result = cached_artifacts_exist("doc1", 2, collection)

    assert result is False
    collection.get.assert_not_called()


def test_cache_missing_embeddings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    image_dir = tmp_path / "data/page_images/doc1"
    image_dir.mkdir(parents=True)

    for page_number in range(1, 3):
        (image_dir / f"page_{page_number:03}.png").write_bytes(b"image")

    collection = MagicMock()
    collection.get.return_value = {"ids": []}

    result = cached_artifacts_exist("doc1", 2, collection)

    assert result is False
    collection.get.assert_called_once_with(
        where={"doc_id": "doc1"},
        include=[],
    )


def test_cache_valid_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    image_dir = tmp_path / "data/page_images/doc1"
    image_dir.mkdir(parents=True)

    for page_number in range(1, 3):
        (image_dir / f"page_{page_number:03}.png").write_bytes(b"image")

    collection = MagicMock()
    collection.get.return_value = {"ids": ["chunk1", "chunk2"]}

    result = cached_artifacts_exist("doc1", 2, collection)

    assert result is True
    collection.get.assert_called_once_with(
        where={"doc_id": "doc1"},
        include=[],
    )

def test_reuses_provided_embedding_model(service_mocks):
    model = MagicMock()
    collection = MagicMock()

    with (
        patch("doubtdesk.ingestion.service.load_embedding_model") as load_model,
        patch("doubtdesk.ingestion.service.get_collection") as get_collection,
        patch("doubtdesk.ingestion.service.embed_chunks") as embed,
    ):
        result = process_pdf(
            Path("biology.pdf"),
            embedding_model=model,
            collection=collection,
        )

    assert result["status"] == "completed"
    load_model.assert_not_called()
    get_collection.assert_not_called()
    embed.assert_called_once_with(["chunk1"], model)