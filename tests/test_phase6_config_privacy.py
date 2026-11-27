from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import Mock


def test_central_runtime_settings_keep_existing_env_names(monkeypatch):
    from doubtdesk.config import runtime_settings, Settings
    monkeypatch.setenv('OUTPUT_JUDGE_ENABLED', 'false')
    monkeypatch.setenv('EVIDENCE_TEXT_THRESHOLD', '0.85')
    monkeypatch.setenv('OUTPUT_JUDGE_RPM', '10')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://app@db.internal:5432/doubtdesk')
    config = runtime_settings()
    assert not config.output_judge_enabled
    assert config.evidence_text_threshold == .85
    assert config.output_judge_rpm == 10
    assert config.database_url == 'postgresql://app@db.internal:5432/doubtdesk'
    assert Settings(_env_file=None, output_judge_enabled=True).output_judge_enabled


def test_legacy_saved_footer_uses_current_owners_filename(tmp_path, monkeypatch):
    from doubtdesk.conversations.routes import get_conversation
    from doubtdesk.memory.store import claim_or_check_conversation, save_exchange
    from doubtdesk.ingestion.registry import register_document, update_document_status, attach_document_to_user
    monkeypatch.chdir(tmp_path)
    doc = str(uuid4())
    chat = uuid4()
    register_document(doc, 'first-user-private.pdf', 'hash')
    update_document_status(doc, 'completed', 1)
    attach_document_to_user('second', doc, 'my-book.pdf')
    claim_or_check_conversation(str(chat), 'second')
    save_exchange(str(chat), 'q', 'A fact [1]', [doc], 'grounded', user_id='second', sources=[{'doc_id':doc, 'page_numbers':[1]}], formatted_answer='A fact [1]\n\nSources:\n- first-user-private.pdf: PDF page(s) 1')
    response = get_conversation(chat, {'user_id':'second'})
    footer = response['messages'][0]['formatted_answer']
    assert 'my-book.pdf' in footer and 'first-user-private.pdf' not in footer
