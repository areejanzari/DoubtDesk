from unittest.mock import Mock
from uuid import uuid4
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from doubtdesk.auth import store, service
from doubtdesk.api.schemas import AskRequest
from doubtdesk.memory.store import claim_or_check_conversation, save_exchange


def test_invalid_reset_does_not_hash(monkeypatch):
    hashed=Mock(); monkeypatch.setattr(store,'hash_password',hashed)
    assert store.reset_password('invalid','new-password') is False
    hashed.assert_not_called()


def test_missing_login_uses_dummy_password_verification(monkeypatch):
    verify=Mock(return_value=True); monkeypatch.setattr(service,'verify_password',verify)
    assert service.authenticate_user('absent','password') is None
    verify.assert_called_once()


def test_proxy_header_only_when_configured(monkeypatch):
    from doubtdesk.auth.dependencies import get_client_ip
    request=SimpleNamespace(client=SimpleNamespace(host='10.0.0.2'),headers={'x-forwarded-for':'198.51.100.99, 203.0.113.4, 10.0.0.1'})
    monkeypatch.delenv('TRUSTED_PROXY_COUNT',raising=False)
    assert get_client_ip(request)=='10.0.0.2'
    monkeypatch.setenv('TRUSTED_PROXY_COUNT','2')
    assert get_client_ip(request)=='203.0.113.4'
    request.headers={'x-forwarded-for':'not-an-ip'}
    assert get_client_ip(request)=='10.0.0.2'


def test_atomic_action_rate_limit_per_email_and_ip():
    from doubtdesk.auth.store import consume_auth_attempt
    for _ in range(5): assert consume_auth_attempt('signup','ip','person@example.com')
    assert not consume_auth_attempt('signup','other-ip','PERSON@example.com')
    assert consume_auth_attempt('reset','ip','person@example.com')
    for n in range(15): assert consume_auth_attempt('signup','ip',f'x{n}@example.com')
    assert not consume_auth_attempt('signup','ip','fresh@example.com')


def test_save_requires_owner():
    claim_or_check_conversation('chat','alice')
    with pytest.raises(TypeError): save_exchange('chat','question','answer',[],'grounded')
    with pytest.raises(ValueError): save_exchange('chat','question','answer',[],'grounded',user_id='bob')


def test_request_limits_and_deduplication():
    doc=str(uuid4())
    assert AskRequest(question='q',doc_ids=[doc,doc]).doc_ids==[doc]
    for ids in [['bad-id'],[str(uuid4()) for _ in range(11)]]:
        with pytest.raises(ValidationError): AskRequest(question='q',doc_ids=ids)
    with pytest.raises(ValidationError): AskRequest(question='x'*2001)
