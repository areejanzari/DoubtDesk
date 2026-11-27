from unittest.mock import Mock

import numpy as np

from doubtdesk.db import EMBEDDING_DIMENSIONS

from doubtdesk.memory.store import claim_or_check_conversation, save_exchange


def test_history_combines_similarity_and_recency_without_crossing_owners():
    from doubtdesk.memory.context import build_history
    claim_or_check_conversation('chat','alice')
    claim_or_check_conversation('other','bob')
    ids = []
    for i, topic in enumerate(['biology','physics','biology','physics','chemistry']):
        ids.append(save_exchange('chat', f'{topic} {i}', 'A'*1500, [topic], 'grounded', user_id='alice'))
    skipped = save_exchange('chat','unavailable','No evidence',['biology'],'general_knowledge',user_id='alice',memory_eligible=False)
    foreign = save_exchange('other','private','private',['biology'],'grounded',user_id='bob')
    search = Mock(return_value=[{'exchange_id':x, 'question':'untrusted vector data','answer_excerpt':'ignored'} for x in [ids[0],ids[2],ids[4],skipped,foreign]])
    result=build_history('chat','alice','biology',['biology'],model=Mock(),search=search)
    assert [x['question'] for x in result] == ['biology 0','biology 2','physics 3','chemistry 4']
    assert [len(x['answer_excerpt']) for x in result] == [300,300,300,1200]
    assert search.call_args.kwargs['exclude_ids'] == {ids[3],ids[4]}
    assert build_history('chat','bob','anything',[]) == []


def at_distance(distance):
    vector = np.zeros(EMBEDDING_DIMENSIONS)
    vector[0] = 1 - distance
    vector[1] = (1 - (1 - distance) ** 2) ** 0.5
    return vector


def test_distance_and_exclude_ids():
    from doubtdesk.memory.vector import add_exchange_memory, search_memory
    claim_or_check_conversation('chat','alice')
    claim_or_check_conversation('other','bob')
    excluded = save_exchange('chat','excluded','answer',['bio'],'grounded',user_id='alice')
    too_far = save_exchange('chat','too far','answer',['bio'],'grounded',user_id='alice')
    match = save_exchange('chat','match','answer',['bio'],'grounded',user_id='alice')
    other_doc = save_exchange('chat','other doc','answer',['chem'],'grounded',user_id='alice')
    foreign = save_exchange('other','foreign','answer',['bio'],'grounded',user_id='bob')
    for exchange_id, distance in [(excluded,.1),(too_far,.7),(match,.3),(other_doc,.05),(foreign,0.)]:
        add_exchange_memory(exchange_id=exchange_id, question='q', answer='a', model=Mock(encode=Mock(return_value=np.array([at_distance(distance)]))))
    model=Mock(encode=Mock(return_value=np.array([at_distance(0.)])))
    results=search_memory(conversation_id='chat',user_id='alice',question='question',doc_ids=['bio'],model=model,exclude_ids={excluded},max_distance=.6)
    assert [r['exchange_id'] for r in results]==[match]
    assert search_memory(conversation_id='chat',user_id='bob',question='question',doc_ids=['bio'],model=model) == []
