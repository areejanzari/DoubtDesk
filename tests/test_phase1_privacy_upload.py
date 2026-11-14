import sqlite3
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pymupdf
import pytest

from doubtdesk.ingestion.registry import (
    initialize_registry, register_document, update_document_status,
    attach_document_to_user, get_document_names,
)
from doubtdesk.answering.formatting import format_answer
from doubtdesk.ingestion.extractor import extract_page_content


def test_shared_hash_uses_only_requesting_users_filename(tmp_path):
    db = tmp_path / 'docs.db'
    initialize_registry(db)
    register_document('shared', 'private-a.pdf', 'same-hash', db)
    update_document_status('shared', 'completed', 1, db)
    attach_document_to_user('a', 'shared', 'private-a.pdf', db)
    attach_document_to_user('b', 'shared', 'private-b.pdf', db)
    answer = {'status': 'grounded', 'answer': 'A fact [1]', 'sources': [{'doc_id': 'shared', 'page_numbers': [1]}]}
    with sqlite3.connect(db) as conn:
        for user, own, other in [('a', 'private-a.pdf', 'private-b.pdf'), ('b', 'private-b.pdf', 'private-a.pdf')]:
            formatted = format_answer(answer, get_document_names(conn, ['shared'], user_id=user))
            assert own in formatted and other not in formatted
        assert get_document_names(conn, ['shared'], user_id='stranger') == {}


def test_ocr_failure_preserves_native_text():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), 'A native fact')
        with patch('doubtdesk.ingestion.extractor.classify_page', return_value='ocr_candidate'), patch('doubtdesk.ingestion.extractor.extract_text_with_ocr', side_effect=RuntimeError('OCR unavailable')):
            assert 'A native fact' in extract_page_content(page)['text']


def test_atomic_hash_claim(tmp_path):
    from doubtdesk.ingestion.registry import claim_document_ingestion, DocumentProcessing
    db = tmp_path / 'docs.db'
    initialize_registry(db)
    def claim(i):
        try:
            return claim_document_ingestion(str(i), 'same.pdf', 'same-hash', db_path=db)
        except DocumentProcessing:
            return None
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(claim, range(4)))
    assert sum(x is not None for x in results) == 1
