from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def http(tmp_path, monkeypatch):
    from doubtdesk.api import routes
    from doubtdesk.auth.dependencies import require_current_user
    from doubtdesk.memory.store import initialize_memory_db
    from doubtdesk.ingestion.registry import initialize_registry
    monkeypatch.chdir(tmp_path)
    initialize_registry()
    initialize_memory_db()
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_current_user] = lambda: {'user_id':'u'}
    app.state.embedding_model = Mock()
    app.state.collection = Mock()
    app.state.graph = Mock()
    app.state.preparation_graph = Mock()
    with TestClient(app) as client:
        yield client, app, routes


@pytest.mark.parametrize('kind', ['encrypted','page-limit','corrupt'])
def test_pdf_validation_rejects_invalid_inputs(http, monkeypatch, kind):
    client, app, routes = http
    process = Mock()
    monkeypatch.setattr(routes, 'process_pdf', process)
    if kind == 'corrupt':
        payload = b'%PDF-invalid-content'
    else:
        with pymupdf.open() as pdf:
            pdf.new_page()
            if kind == 'encrypted':
                payload = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='secret')
            else:
                pdf.new_page()
                monkeypatch.setenv('MAX_PDF_PAGES', '1')
                payload = pdf.tobytes()
    result = client.post('/documents/upload', files={'file':('test.pdf',payload,'application/pdf')})
    assert result.status_code == 400
    process.assert_not_called()
    assert {'encrypted':'Encrypted', 'page-limit':'page limit','corrupt':'corrupted'}[kind].lower() in result.json()['detail'].lower()


def test_processing_document_returns_conflict(http, monkeypatch):
    client, _, routes = http
    from doubtdesk.ingestion.registry import DocumentProcessing
    monkeypatch.setattr(routes, 'process_pdf', Mock(side_effect=DocumentProcessing('Still processing')))
    with pymupdf.open() as pdf:
        pdf.new_page()
        payload=pdf.tobytes()
    result=client.post('/documents/upload', files={'file':('test.pdf',payload,'application/pdf')})
    assert result.status_code==409 and 'processing' in result.json()['detail']


def test_failed_request_cleans_only_automatic_chat(http):
    client, app, _ = http
    from doubtdesk.memory.store import claim_or_check_conversation, list_user_conversations
    explicit=str(uuid4())
    claim_or_check_conversation(explicit,'u')
    for endpoint, graph in [('/ask',app.state.graph),('/ask/stream',app.state.preparation_graph)]:
        graph.invoke.side_effect=ValueError('internal')
        response=client.post(endpoint,json={'question':'question'})
        if endpoint=='/ask': assert response.status_code==500
        else: assert 'event: error' in response.text
        assert [row['conversation_id'] for row in list_user_conversations('u')]==[explicit]


def test_pdf_validation_releases_mapping_on_corrupt_upload(http, monkeypatch, tmp_path):
    from tempfile import TemporaryDirectory

    client, _, routes = http
    original_open = pymupdf.open
    mappings = []
    views = []
    process = Mock()
    monkeypatch.setattr(routes, 'process_pdf', process)
    monkeypatch.setattr(routes, 'TemporaryDirectory', lambda: TemporaryDirectory(dir=tmp_path))

    def observe_open(*args, **kwargs):
        view = kwargs.get('stream')
        assert isinstance(view, memoryview), 'Validation must not give MuPDF ownership of a temporary file handle'
        views.append(view)
        mappings.append(view.obj)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(routes.pymupdf, 'open', observe_open)
    result = client.post('/documents/upload', files={'file': ('broken.pdf', b'%PDF-invalid-content', 'application/pdf')})
    assert result.status_code == 400
    assert result.json()['detail'] == 'PDF is corrupted or unreadable.'
    process.assert_not_called()
    assert len(mappings) == 1 and mappings[0].closed
    with pytest.raises(ValueError, match='released memoryview'):
        len(views[0])
    assert not list(tmp_path.rglob('broken.pdf'))
