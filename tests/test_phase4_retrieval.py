from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
from doubtdesk.ingestion.schemas import ChunkRecord
from doubtdesk.retrieval.lexical_store import replace_document_chunks, lexical_candidates


def test_stopwords_do_not_create_irrelevant_matches(tmp_path):
    db=tmp_path/'fts.db'
    replace_document_chunks('d',[ChunkRecord(doc_id='d',page_number=1,chunk_index=0,text='the is a'),ChunkRecord(doc_id='d',page_number=2,chunk_index=0,text='photosynthesis converts energy')],db)
    assert [r['page_number'] for r in lexical_candidates(['what is the photosynthesis'],['d'],db_path=db)[0]] == [2]
    assert lexical_candidates(['the is'],['d'],db_path=db)[0][0]['page_number']==1
    with patch('doubtdesk.retrieval.lexical_store.initialize_lexical_index', side_effect=AssertionError('DDL in query')):
        assert lexical_candidates(['photosynthesis'],['d'],db_path=db)[0]


def test_dynamic_scope_batches_and_matches_fixed_batch():
    from doubtdesk.guardrails.scope import ScopeClassifier
    class Tokenizer:
        def __call__(self, system, questions, **kwargs):
            questions = [questions] if isinstance(questions,str) else questions
            values=np.array([[len(q),0] for q in questions])
            return {'input_ids':values,'attention_mask':np.ones_like(values)}
    def instance(batch):
        obj=ScopeClassifier.__new__(ScopeClassifier);obj.tokenizer=Tokenizer();obj.session=Mock()
        obj.session.get_inputs.return_value=[SimpleNamespace(name='ids',shape=[batch,'seq']),SimpleNamespace(name='mask',shape=[batch,'seq'])]
        obj.session.run.side_effect=lambda _,inputs:[np.array([[float(row[0]),5.] for row in inputs['ids']])]
        return obj
    dynamic=instance('batch');fixed=instance(1)
    questions=['abc','abcdef','a']
    assert dynamic.predict(questions)==fixed.predict(questions)
    assert dynamic.session.run.call_count==1
    assert fixed.session.run.call_count==3
