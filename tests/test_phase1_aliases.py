from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from doubtdesk.answering.vision import build_grounded_content, generate_vision_answer

DOC = 'e8be2d4d-9094-43e8-8cec-7fe4e7853cce'
OTHER = '1c2f3e40-1111-2222-3333-444455556666'
CHUNKS = [{'doc_id': DOC, 'page_number': 1, 'text': 'Fact A'}, {'doc_id': OTHER, 'page_number': 1, 'text': 'Fact B'}]


def test_aliases_map_by_document_and_page_not_by_page_number_alone():
    provider = Mock()
    provider.responses.parse.return_value = SimpleNamespace(output_parsed={'status':'sufficient','answer':'A (S2) B (S1)','sources':['S2','S1']}, status='completed', usage=None)
    evidence = generate_vision_answer(provider, 'fake', build_grounded_content('Question?', CHUNKS, []))
    assert evidence.answer == 'A [1] B [2]'
    assert [s.model_dump() for s in evidence.sources] == [{'doc_id':OTHER,'page_numbers':[1]}, {'doc_id':DOC,'page_numbers':[1]}]
    call = provider.responses.parse.call_args.kwargs
    assert DOC not in str(call['input']) and OTHER not in str(call['input'])
    assert call['max_output_tokens'] > 300


@pytest.mark.parametrize('answer,sources', [('Unknown (S9)', ['S9']), ('Unlisted (S2)', ['S1'])])
def test_invalid_alias_is_a_failed_candidate(answer, sources):
    from doubtdesk.answering.vision import CandidateValidationError
    provider = Mock()
    provider.responses.parse.return_value = SimpleNamespace(output_parsed={'status':'sufficient','answer':answer,'sources':sources}, status='completed', usage=None)
    with pytest.raises(CandidateValidationError):
        generate_vision_answer(provider,'fake',build_grounded_content('Question?',CHUNKS,[]))


def test_incomplete_response_is_a_failed_candidate():
    from doubtdesk.answering.vision import CandidateValidationError
    provider = Mock()
    provider.responses.parse.return_value = SimpleNamespace(output_parsed=None,status='incomplete',usage=None)
    with pytest.raises(CandidateValidationError):
        generate_vision_answer(provider,'fake',build_grounded_content('Question?',CHUNKS,[]))


@pytest.mark.parametrize('always_invalid', [False, True])
def test_stream_invalid_citation_regenerates_before_delivering(always_invalid):
    import json
    from doubtdesk.api.streaming import stream_answer
    from doubtdesk.answering.output_guardrails import JudgeVerdict

    class FakeStream:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def __iter__(self):
            yield SimpleNamespace(type='response.output_text.delta', delta=json.dumps({'status':'sufficient','answer':'INVALID (S9)','sources':['S9']}))
        def get_final_response(self): return SimpleNamespace(status='completed',usage=None)

    provider = Mock()
    provider.responses.stream.return_value = FakeStream()
    provider.responses.parse.return_value = SimpleNamespace(status='completed', usage=None, output_parsed={'status':'sufficient','answer':'INVALID (S9)' if always_invalid else 'A fact (S1)','sources':['S9' if always_invalid else 'S1']})
    judge = Mock()
    judge.evaluate.return_value = JudgeVerdict(passed=True,relevance=1,groundedness=1,completeness=1,safety='pass')
    prepared = {'question':'According to this PDF, what is the fact?', 'doc_ids':[DOC], 'retrieved_chunks':CHUNKS, 'evidence_pages':CHUNKS,'evidence_mode':'text'}
    iterator = stream_answer(prepared,provider,'fake',judge=judge)
    events=[]
    while True:
        try: events.append(next(iterator))
        except StopIteration as stop:
            result=stop.value
            break
    text=''.join(data['text'] for event,data in events if event=='token')
    assert 'INVALID' not in text
    assert provider.responses.parse.call_count == 1
    if always_invalid:
        assert 'couldn\'t find enough evidence' in text
        assert judge.evaluate.call_count == 0
    else:
        assert text == 'A fact [1]'
        assert result['status']=='grounded'
        assert judge.evaluate.call_count == 1
