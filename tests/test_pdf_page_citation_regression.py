import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from doubtdesk.answering.formatting import normalize_answer_citations, format_answer

DOC = 'e8be2d4d-9094-43e8-8cec-7fe4e7853cce'
OTHER = '1c2f3e40-1111-2222-3333-444455556666'
SOURCES = [{'doc_id': DOC, 'page_numbers': [1]}]


@pytest.mark.parametrize('citation', ['(PDF p. 1)', '(PDF page 1)', '(pdf p 1)', f'({DOC}, p. 1)'])
def test_screenshot_citation(citation):
    answer = f'Ureotelic animals mainly excrete urea. {citation}'
    assert normalize_answer_citations(answer, SOURCES) == 'Ureotelic animals mainly excrete urea. [1]'


def test_page_number_is_not_citation_index():
    sources = [{'doc_id': DOC, 'page_numbers': [7, 2, 7]}, {'doc_id': OTHER, 'page_numbers': [4]}]
    assert normalize_answer_citations('A (PDF p. 2) B (PDF page 4) C (PDF p. 7)', sources) == 'A [2] B [3] C [1]'


def test_ambiguous_or_unknown_page_never_links_wrong_document():
    sources = SOURCES + [{'doc_id': OTHER, 'page_numbers': [1]}]
    assert normalize_answer_citations('A (PDF p. 1) B (PDF p. 2)', sources) == 'A [ambiguous source] B [unverified source]'
    assert normalize_answer_citations(f'A ({OTHER}, p. 1)', sources) == 'A [2]'


def test_normalizing_twice_preserves_numbered_citations():
    normalized = normalize_answer_citations('A (PDF p. 1)', SOURCES)
    assert normalize_answer_citations(normalized, SOURCES) == normalized


def test_general_not_rewritten():
    result = {'status': 'general_knowledge', 'answer': 'A (PDF p. 1)', 'sources': []}
    assert format_answer(result, {}) == 'A (PDF p. 1)\n\nNot from document.'


@pytest.mark.parametrize('mode', ['text', 'visual'])
@pytest.mark.parametrize('use_judge', [True, False])
def test_streamed_grounded_citation_matches_complete_payload(mode, use_judge, tmp_path, monkeypatch):
    from doubtdesk.api import streaming
    from doubtdesk.answering.output_guardrails import JudgeVerdict
    from doubtdesk.api.schemas import AskResponse

    raw_answer = 'Ureotelic animals mainly excrete urea. (PDF p. 1)'
    raw = json.dumps({'status': 'sufficient', 'answer': raw_answer, 'sources': ['S1']})

    class FakeStream:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def __iter__(self):
            for pos in range(0, len(raw), 5):
                yield SimpleNamespace(type='response.output_text.delta', delta=raw[pos:pos+5])
        def get_final_response(self): return SimpleNamespace(output_text=raw, usage=None)

    provider = Mock()
    provider.responses.stream.return_value = FakeStream()
    judge = Mock()
    judge.evaluate.return_value = JudgeVerdict(passed=True, relevance=1, groundedness=1, completeness=1, safety='pass')
    page = {'doc_id': DOC, 'page_number': 1}
    prepared = {'question': 'Ureotelic animals?', 'doc_ids': [DOC], 'evidence_mode': mode,
                'retrieved_chunks': [{**page, 'text': raw_answer}], 'retrieved_pages': [], 'evidence_pages': [page]}
    if mode == 'visual':
        image = tmp_path / 'page.png'
        image.write_bytes(b'fake-png')
        prepared['retrieved_pages'] = [{**page, 'image_path': str(image)}]
    monkeypatch.setattr(streaming, 'get_document_names', lambda *_, **kwargs: {DOC: 'kebo116.pdf'})
    iterator = streaming.stream_answer(prepared, provider, 'fake-model', judge=judge if use_judge else None)
    tokens = []
    while True:
        try:
            event, data = next(iterator)
            if event == 'token': tokens.append(data['text'])
        except StopIteration as finished:
            result = finished.value
            break
    complete = AskResponse.model_validate(streaming.finalize_stream_result(result, str(tmp_path / 'docs.db'), user_id='u'))
    assert ''.join(tokens) == complete.answer == 'Ureotelic animals mainly excrete urea. [1]'
    assert complete.model_dump()['sources'] == SOURCES
    assert 'kebo116.pdf: PDF page(s) 1' in complete.formatted_answer
    assert provider.responses.stream.call_count == 1
    assert judge.evaluate.call_count == int(use_judge)
