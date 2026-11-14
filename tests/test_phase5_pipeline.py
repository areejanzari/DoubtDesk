from types import SimpleNamespace
from unittest.mock import Mock


def test_off_topic_emits_exact_answer():
    from doubtdesk.api.streaming import stream_answer
    provider = Mock()
    events = list(stream_answer({'question': 'Movies?', 'is_off_topic': True, 'answer': 'Ask an academic question.'}, provider, 'fake'))
    assert events and ''.join(data['text'] for event, data in events if event == 'token') == 'Ask an academic question.'
    provider.responses.stream.assert_not_called()


def test_empty_automatic_chat_cleanup_preserves_nonempty_and_renamed(tmp_path):
    from doubtdesk.memory.store import initialize_memory_db, claim_or_check_conversation, delete_empty_conversation, list_user_conversations, save_exchange, rename_user_conversation
    db = tmp_path / 'history.db'
    initialize_memory_db(db)
    for chat in ('empty', 'saved', 'renamed', 'other'):
        assert claim_or_check_conversation(chat, 'other' if chat == 'other' else 'user', db_path=db)
    save_exchange(conversation_id='saved', question='q', answer='a', doc_ids=[], status='general_knowledge', user_id='user', db_path=db)
    rename_user_conversation('renamed', 'user', 'Keep this', db_path=db)
    assert delete_empty_conversation('empty', 'user', db_path=db)
    assert not delete_empty_conversation('saved', 'user', db_path=db)
    assert not delete_empty_conversation('renamed', 'user', db_path=db)
    assert not delete_empty_conversation('other', 'user', db_path=db)
    assert {row['conversation_id'] for row in list_user_conversations('user', db_path=db)} == {'saved', 'renamed'}


def test_shared_nonstream_runner_calls_same_answer_generator(monkeypatch):
    from doubtdesk.api import streaming
    prep = Mock()
    prep.invoke.return_value = {'question':'q','user_id':'u', 'evidence_average':.9}
    def answer(state, *args, **kwargs):
        assert state['user_id'] == 'u'
        yield ('token', {'text':'answer'})
        return {'answer':'answer','status':'general_knowledge','sources':[]}
    monkeypatch.setattr(streaming, 'stream_answer', answer)
    monkeypatch.setattr(streaming, 'finalize_stream_result', lambda result, **kwargs: {**result, 'formatted_answer':'answer'})
    graph = streaming.PreparedAnswerGraph(prep, object(), 'fake', object())
    result = graph.invoke({'question':'q','user_id':'u'}, config={'callbacks':[]})
    assert result['answer'] == 'answer' and result['evidence_average'] == .9
    prep.invoke.assert_called_once_with({'question':'q','user_id':'u'},config={'callbacks':[]})


import json
from uuid import uuid4
import pytest


@pytest.mark.parametrize('scenario', ['grounded', 'insufficient', 'document-specific', 'repair', 'fallback', 'fallback-error', 'unavailable'])
def test_both_http_routes_share_answer_behavior(scenario, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from doubtdesk.api import routes, streaming
    from doubtdesk.auth.dependencies import require_current_user
    from doubtdesk.answering.output_guardrails import JudgeVerdict
    from doubtdesk.ingestion.registry import initialize_registry, register_document, update_document_status, attach_document_to_user
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()
    db = tmp_path / 'data/documents.db'
    initialize_registry(db)
    doc = str(uuid4())
    register_document(doc, 'private.pdf', 'hash', db)
    update_document_status(doc, 'completed', 1, db)
    attach_document_to_user('u', doc, 'own.pdf', db)
    monkeypatch.setattr(routes, 'claim_or_check_conversation', lambda **kw: True)
    monkeypatch.setattr(routes, '_save_exchange_and_memory', Mock())
    monkeypatch.setattr(routes, 'graph_config', lambda: None)
    question = 'According to this PDF, what is the fact?' if scenario == 'document-specific' else 'What is photosynthesis?'
    chunks = [] if scenario == 'document-specific' else [{'doc_id':doc, 'page_number':1,'text':'A fact'}]
    state = {'question':question,'doc_ids':[doc],'user_id':'u','retrieved_chunks':chunks,'retrieved_pages':[], 'evidence_pages':chunks, 'evidence_mode':'text' if chunks else 'low'}
    def approved(ok):
        return JudgeVerdict(passed=ok,relevance=1 if ok else .1,groundedness=1,completeness=1,safety='pass',feedback='Fix it')
    class Stream:
        def __init__(self, text): self.text=text
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def __iter__(self): yield SimpleNamespace(type='response.output_text.delta',delta=self.text)
        def get_final_response(self): return SimpleNamespace(status='completed',usage=None,output_text=self.text)
    def create_app():
        provider = Mock()
        raw = json.dumps({'status':'insufficient','answer':'','sources':[]}) if scenario=='insufficient' else json.dumps({'status':'sufficient','answer':'A fact (S1)','sources':['S1']})
        def stream(**kwargs):
            if 'text_format' not in kwargs and scenario=='fallback-error': raise RuntimeError('provider failed')
            return Stream(raw if 'text_format' in kwargs else 'General fact')
        provider.responses.stream.side_effect=stream
        provider.responses.parse.return_value=SimpleNamespace(output_parsed={'status':'sufficient','answer':'Repaired fact (S1)','sources':['S1']},status='completed',usage=None)
        judge=Mock()
        if scenario=='unavailable':
            error=RuntimeError('unavailable'); error.code=503; judge.evaluate.side_effect=error
        else:
            judge.evaluate.side_effect=iter([approved(False),approved(True)] if scenario=='repair' else [approved(False),approved(False),approved(True)] if scenario in ('fallback','fallback-error') else [approved(True)])
        prep=SimpleNamespace(invoke=lambda input,config=None:{**input,**state})
        app=FastAPI(); app.include_router(routes.router)
        app.dependency_overrides[require_current_user]=lambda:{'user_id':'u'}
        app.state.llm_client=provider; app.state.model_name='fake'; app.state.output_judge=judge
        app.state.preparation_graph=prep; app.state.graph=streaming.PreparedAnswerGraph(prep,provider,'fake',judge)
        return app
    payload={'question':question,'doc_ids':[doc],'conversation_id':str(uuid4())}
    with TestClient(create_app()) as http:
        normal=http.post('/ask',json=payload)
    with TestClient(create_app()) as http:
        streamed=http.post('/ask/stream',json=payload)
    assert normal.status_code==streamed.status_code==200, (normal.text,streamed.text)
    blocks=[block.split('\ndata: ',1) for block in streamed.text.split('\n\n') if '\ndata: ' in block]
    final=next(json.loads(data) for event,data in blocks if event=='event: complete')
    assert final==normal.json()
    assert ''.join(json.loads(data)['text'] for event,data in blocks if event=='event: token')==final['answer']
    assert final['status']== {'grounded':'grounded','insufficient':'general_knowledge','document-specific':'general_knowledge','repair':'grounded','fallback':'general_knowledge','fallback-error':'verification_failed','unavailable':'verification_failed'}[scenario]
    if final['status']=='grounded': assert 'own.pdf' in final['formatted_answer'] and 'private.pdf' not in final['formatted_answer']
