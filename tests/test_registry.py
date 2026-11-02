import pytest
import sqlite3
from contextlib import closing
from doubtdesk.ingestion.registry import (
    initialize_registry,
    register_document,
    get_document_by_hash,
)
from doubtdesk.ingestion.registry import (
    update_document_status,
    is_document_cached,
)
from doubtdesk.ingestion.registry import (
    validate_document_ids,
    get_document_page_counts,
    get_document_names,
    list_documents,
    update_document_filename,
    attach_document_to_user,
    validate_user_document_ids,
    user_can_access_document,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test_documents.db"
    initialize_registry(path)
    return path

@pytest.fixture
def populated_db(db_path):
    register_document("doc_001", "biology.pdf", "hash_001", db_path)
    register_document("doc_002", "chemistry.pdf", "hash_002", db_path)
    register_document("doc_003", "physics.pdf", "hash_003", db_path)

    update_document_status("doc_001", "completed", 8, db_path)
    update_document_status("doc_002", "completed", 12, db_path)

    return db_path

def test_register_document(db_path):
    inserted = register_document(
        doc_id="doc_001",
        filename="biology.pdf",
        sha256="hash_001",
        db_path=db_path,
    )

    document = get_document_by_hash("hash_001", db_path)

    assert inserted is True
    assert document["doc_id"] == "doc_001"
    assert document["filename"] == "biology.pdf"
    assert document["status"] == "pending"
    assert document["page_count"] == 0


def test_duplicate_document(db_path):
    register_document("doc_001", "biology.pdf", "hash_001", db_path)

    inserted = register_document(
        "doc_002",
        "biology_copy.pdf",
        "hash_001",
        db_path,
    )

    assert inserted is False


def test_document_not_found(db_path):
    document = get_document_by_hash("nonexistent_hash", db_path)

    assert document is None

def test_update_document_status(db_path):
    register_document("doc_001", "biology.pdf", "hash_001", db_path)

    update_document_status("doc_001", "completed", 8, db_path)

    document = get_document_by_hash("hash_001", db_path)

    assert document["status"] == "completed"
    assert document["page_count"] == 8


@pytest.mark.parametrize("status", ["invalid", "done", "success"])
def test_invalid_status(db_path, status):
    register_document("doc_001", "biology.pdf", "hash_001", db_path)

    with pytest.raises(ValueError, match="Invalid document status"):
        update_document_status("doc_001", status, 8, db_path)


def test_negative_page_count(db_path):
    register_document("doc_001", "biology.pdf", "hash_001", db_path)

    with pytest.raises(ValueError, match="Page count cannot be negative"):
        update_document_status("doc_001", "completed", -1, db_path)


def test_update_nonexistent_document(db_path):
    with pytest.raises(ValueError, match="Document not found"):
        update_document_status("missing_doc", "completed", 8, db_path)


def test_document_cache(db_path):
    register_document("doc_001", "biology.pdf", "hash_001", db_path)

    assert is_document_cached("hash_001", db_path) is False

    update_document_status("doc_001", "completed", 8, db_path)

    assert is_document_cached("hash_001", db_path) is True
    assert is_document_cached("unknown_hash", db_path) is False
    

def test_get_document_page_counts(populated_db):
    with sqlite3.connect(populated_db) as conn:
        result = get_document_page_counts(
            conn, ["doc_001", "doc_002", "doc_003"]
        )

    assert result == {"doc_001": 8, "doc_002": 12}


def test_get_document_names(populated_db):
    with sqlite3.connect(populated_db) as conn:
        result = get_document_names(
            conn, ["doc_001", "doc_002", "doc_003"]
        )

    assert result == {
        "doc_001": "biology.pdf",
        "doc_002": "chemistry.pdf",
    }


def test_validate_document_ids(populated_db):
    with sqlite3.connect(populated_db) as conn:
        validate_document_ids(conn, ["doc_001", "doc_002"])


@pytest.mark.parametrize("doc_id", ["doc_003", "missing_doc"])
def test_invalid_document_ids(populated_db, doc_id):
    with sqlite3.connect(populated_db) as conn:
        with pytest.raises(
            ValueError, match="Invalid or incomplete document IDs"
        ):
            validate_document_ids(conn, [doc_id])


def test_list_documents(populated_db):
    attach_document_to_user("user_A", "doc_001", "biology.pdf", populated_db)
    attach_document_to_user("user_A", "doc_002", "chemistry.pdf", populated_db)
    attach_document_to_user("user_B", "doc_002", "my-chemistry.pdf", populated_db)

    with closing(sqlite3.connect(populated_db)) as conn:
        user_a = list_documents(conn, user_id="user_A")
        user_b = list_documents(conn, user_id="user_B")
        nobody = list_documents(conn, user_id="user_C")

    assert [doc["filename"] for doc in user_a] == [
        "biology.pdf",
        "chemistry.pdf",
    ]
    assert [doc["filename"] for doc in user_b] == ["my-chemistry.pdf"]
    assert nobody == []


def test_validate_user_document_ids_enforces_every_id(populated_db):
    attach_document_to_user("user_A", "doc_001", "biology.pdf", populated_db)
    attach_document_to_user("user_B", "doc_002", "chemistry.pdf", populated_db)

    with closing(sqlite3.connect(populated_db)) as conn:
        validate_user_document_ids(conn, "user_A", ["doc_001"])
        validate_user_document_ids(conn, "user_A", [])
        with pytest.raises(ValueError, match="unavailable"):
            validate_user_document_ids(conn, "user_A", ["doc_001", "doc_002"])
        with pytest.raises(ValueError, match="unavailable"):
            validate_user_document_ids(conn, "user_A", ["doc_003"])
        with pytest.raises(ValueError, match="unavailable"):
            validate_user_document_ids(conn, "user_A", ["missing_doc"])


def test_page_image_document_ownership(populated_db):
    attach_document_to_user("user_A", "doc_001", "biology.pdf", populated_db)

    with closing(sqlite3.connect(populated_db)) as conn:
        assert user_can_access_document(conn, "user_A", "doc_001", 1)
        assert user_can_access_document(conn, "user_A", "doc_001", 8)
        assert not user_can_access_document(conn, "user_A", "doc_001", 9)
        assert not user_can_access_document(conn, "user_B", "doc_001", 1)
        assert not user_can_access_document(conn, "user_A", "doc_003", 1)


def test_update_document_filename(populated_db):
    update_document_filename(
        "doc_001", "updated_biology.pdf", populated_db
    )

    document = get_document_by_hash("hash_001", populated_db)

    assert document["filename"] == "updated_biology.pdf"