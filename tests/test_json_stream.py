import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from doubtdesk.answering.json_stream import AliasRewriter, AnswerFieldStreamer, citation_keys_in_order
from doubtdesk.answering.output_guardrails import JudgeVerdict
from doubtdesk.answering.vision import decode_model_evidence

DOC = "e8be2d4d-9094-43e8-8cec-7fe4e7853cce"
OTHER = "1c2f3e40-1111-2222-3333-444455556666"


def feed_in_pieces(text, size):
    streamer = AnswerFieldStreamer()
    out = "".join(streamer.feed(text[i:i + size]) for i in range(0, len(text), size))
    return streamer, out


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_streamer_decodes_answer_with_escapes_split_anywhere(size):
    answer = 'Line "one"\nTab\there \\ slash / unicode é and emoji 😀 and °C'
    raw = json.dumps({"status": "sufficient", "answer": answer, "sources": ["S1"]})
    streamer, out = feed_in_pieces(raw, size)
    assert out == answer
    assert streamer.status == "sufficient"
    assert streamer.answer_complete


def test_streamer_handles_ascii_escaped_surrogates():
    raw = json.dumps({"status": "sufficient", "answer": "x 😀 y", "sources": []}, ensure_ascii=True)
    assert "\\ud83d" in raw
    _, out = feed_in_pieces(raw, 1)
    assert out == "x 😀 y"


def test_streamer_ignores_other_fields_and_nested_values():
    raw = '{"status":"insufficient","meta":{"answer":"nested"},"sources":["answer"],"answer":""}'
    streamer, out = feed_in_pieces(raw, 4)
    assert out == ""
    assert streamer.status == "insufficient"


def test_streamer_reports_status_before_answer_text():
    streamer = AnswerFieldStreamer()
    assert streamer.feed('{"status":"suff') == ""
    assert streamer.status is None
    assert streamer.feed('icient","answer":"Hel') == "Hel"
    assert streamer.status == "sufficient"


ALIASES = {"S1": (DOC, 3), "S2": (OTHER, 5), "S3": (DOC, 7)}


@pytest.mark.parametrize("size", [1, 2, 5, 100])
def test_alias_rewriter_numbers_by_first_appearance_across_splits(size):
    text = "First (S2) then ( S1 ) again (S2) and (S3). Not a cite (see above) or (S99)."
    rewriter = AliasRewriter(ALIASES)
    out = "".join(rewriter.feed(text[i:i + size]) for i in range(0, len(text), size)) + rewriter.flush()
    assert out == "First [1] then [2] again [1] and [3]. Not a cite (see above) or (S99)."


def test_alias_rewriter_maps_unique_pdf_page_and_uuid_citations():
    rewriter = AliasRewriter(ALIASES)
    out = rewriter.feed(f"A (PDF p. 7) B ({OTHER}, p. 5) C (PDF page 3)") + rewriter.flush()
    assert out == "A [1] B [2] C [3]"


def test_alias_rewriter_holds_partial_citation_until_closed():
    rewriter = AliasRewriter(ALIASES)
    assert rewriter.feed("Value (S") == "Value "
    assert rewriter.feed("1") == ""
    assert rewriter.feed(") done") == "[1] done"


def test_alias_rewriter_releases_unclosed_parenthesis_on_flush():
    rewriter = AliasRewriter(ALIASES)
    assert rewriter.feed("Open (paren") == "Open "
    assert rewriter.flush() == "(paren"


def test_decoded_sources_follow_inline_order_so_stream_numbers_match():
    payload = {"status": "sufficient", "answer": "B (S2) then A (S1) then C (S3)", "sources": ["S1", "S3", "S2"]}
    evidence = decode_model_evidence(payload, ALIASES)
    flat = [(source.doc_id, page) for source in evidence.sources for page in source.page_numbers]
    assert flat == [(OTHER, 5), (DOC, 3), (DOC, 7)]
    rewriter = AliasRewriter(ALIASES)
    assert rewriter.feed(payload["answer"]) + rewriter.flush() == evidence.answer == "B [1] then A [2] then C [3]"


def test_citation_keys_in_order_skips_ambiguous_pdf_pages():
    aliases = {"S1": (DOC, 1), "S2": (OTHER, 1)}
    assert citation_keys_in_order("x (PDF p. 1) (S2)", aliases) == [(OTHER, 1)]


def test_grounded_tokens_stream_before_judge_check():
    from doubtdesk.api.streaming import stream_answer

    raw = json.dumps({"status": "sufficient", "answer": "Urea is excreted (S1).", "sources": ["S1"]})

    class FakeStream:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def __iter__(self):
            for pos in range(0, len(raw), 3):
                yield SimpleNamespace(type="response.output_text.delta", delta=raw[pos:pos + 3])
        def get_final_response(self): return SimpleNamespace(status="completed", usage=None)

    provider = Mock()
    provider.responses.stream.return_value = FakeStream()
    judge = Mock()
    judge.evaluate.return_value = JudgeVerdict(passed=True, relevance=1, groundedness=1, completeness=1, safety="pass")
    page = {"doc_id": DOC, "page_number": 1}
    prepared = {"question": "Urea?", "doc_ids": [DOC], "evidence_mode": "text", "retrieved_pages": [],
                "retrieved_chunks": [{**page, "text": "Urea is excreted."}], "evidence_pages": [page]}
    iterator = stream_answer(prepared, provider, "fake", judge=judge)
    events = []
    while True:
        try:
            events.append(next(iterator))
        except StopIteration as stop:
            result = stop.value
            break
    kinds = [event for event, _ in events]
    first_token = kinds.index("token")
    assert events[first_token - 1] == ("status", {"stage": "verifying"})
    assert first_token < events.index(("status", {"stage": "checking"}))
    assert "reset" not in kinds
    assert sum(kind == "token" for kind in kinds) > 3
    assert "".join(data["text"] for event, data in events if event == "token") == result["answer"] == "Urea is excreted [1]."


def test_insufficient_grounded_result_never_emits_tokens():
    from doubtdesk.api.streaming import stream_grounded

    raw = json.dumps({"status": "insufficient", "answer": "", "sources": []})

    class FakeStream:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def __iter__(self):
            yield SimpleNamespace(type="response.output_text.delta", delta=raw)
        def get_final_response(self): return SimpleNamespace(status="completed", usage=None)

    provider = Mock()
    provider.responses.stream.return_value = FakeStream()
    chunk = {"doc_id": DOC, "page_number": 1, "text": "Text"}
    assert stream_grounded(provider, "fake", "Q?", [], chunks=[chunk]) is None
