import pytest
from pydantic import ValidationError
from doubtdesk.answering.evidence_selection import TextEvidenceAssessment, choose_evidence_route


@pytest.mark.parametrize('score,tier', [(0.8,'high'),(0.79,'moderate'),(0.5,'moderate'),(0.49,'low')])
def test_one_score_is_independent_of_paraphrase_punctuation(score, tier):
    assessment = TextEvidenceAssessment(sufficiency_score=score)
    chunks = [{'doc_id':'a','page_number':1,'text':'A fact'}]
    result = choose_evidence_route(assessment, ['“Question?”', 'Rewording.','Question'], chunks)
    assert result['evidence_tier'] == tier
    assert result['evidence_average'] == score


@pytest.mark.parametrize('value', [-0.1, 1.1, float('nan'), None])
def test_invalid_score_rejected(value):
    with pytest.raises(ValidationError):
        TextEvidenceAssessment(sufficiency_score=value)
