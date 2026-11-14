from unittest.mock import Mock

from doubtdesk.memory.store import initialize_memory_db, claim_or_check_conversation, save_exchange


def test_history_combines_similarity_and_recency_without_crossing_owners(tmp_path):
    from doubtdesk.memory.context import build_history
    db = tmp_path / 'memory.db'
    initialize_memory_db(db)
    claim_or_check_conversation('chat','alice',db)
    claim_or_check_conversation('other','bob',db)
    ids = []
    for i, topic in enumerate(['biology','physics','biology','physics','chemistry']):
        ids.append(save_exchange('chat', f'{topic} {i}', 'A'*1500, [topic], 'grounded', db, user_id='alice'))
    skipped = save_exchange('chat','unavailable','No evidence',['biology'],'general_knowledge',db,user_id='alice',memory_eligible=False)
    foreign = save_exchange('other','private','private',['biology'],'grounded',db,user_id='bob')
    search = Mock(return_value=[{'exchange_id':x, 'question':'untrusted vector data','answer_excerpt':'ignored'} for x in [ids[0],ids[2],ids[4],skipped,foreign]])
    result=build_history('chat','alice','biology',['biology'],db_path=db,model=Mock(),collection=Mock(),search=search)
    assert [x['question'] for x in result] == ['biology 0','biology 2','physics 3','chemistry 4']
    assert [len(x['answer_excerpt']) for x in result] == [300,300,300,1200]
    assert search.call_args.kwargs['exclude_ids'] == {ids[3],ids[4]}
    assert build_history('chat','bob','anything',[],db_path=db) == []


def test_distance_and_exclude_ids():
    from doubtdesk.memory.vector import search_memory
    model=Mock(); model.encode.return_value.tolist.return_value=[[1.0]]
    collection=Mock(); collection.count.return_value=4
    metadata={'conversation_id':'chat','user_id':'alice','doc_ids':'["bio"]','question':'question','answer_excerpt':'answer'}
    collection.query.return_value={'ids':[['exchange_1','exchange_2','exchange_3','exchange_4']], 'metadatas':[[metadata,metadata,metadata,{**metadata,'user_id':'bob'}]],'distances':[[.1,.7,.3,.1]]}
    results=search_memory(conversation_id='chat',user_id='alice',question='question',doc_ids=['bio'],model=model,collection=collection,exclude_ids={1},max_distance=.6)
    assert [r['exchange_id'] for r in results]==[3]
    collection.count.assert_called_once_with()
