from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from doubtdesk.graph.state import DoubtDeskState
from doubtdesk.graph.nodes import (
    retrieve_node,
    evidence_selection_node,
    paraphrase_node,
    scope_node,
    rejection_node,
    aparaphrase_node,
    ascope_node,
    aretrieve_node,
    aevidence_selection_node,
)
from doubtdesk.graph.routing import route_documents, route_retrieval


def route_scope(state: DoubtDeskState) -> str:
    if state["is_off_topic"]:
        return "reject"
    return route_documents(state)


def build_preparation_graph(
    embedding_model, client, model_name, scope_classifier,
    paraphrase_model_name=None, assessment_model_name=None, reranker=None, paraphrase_client=None,
):
    workflow = StateGraph(DoubtDeskState)
    paraphrase_model = paraphrase_model_name or model_name
    assessment_model = assessment_model_name or model_name
    workflow.add_node("paraphrase", RunnableLambda(
        paraphrase_node(paraphrase_client or client, paraphrase_model, embedding_model),
        afunc=aparaphrase_node(paraphrase_client or client, paraphrase_model, embedding_model),
    ))
    workflow.add_node("scope", RunnableLambda(scope_node(scope_classifier), afunc=ascope_node(scope_classifier)))
    workflow.add_node("reject", rejection_node)
    workflow.add_node("retrieve", RunnableLambda(
        retrieve_node(embedding_model, reranker), afunc=aretrieve_node(embedding_model, reranker),
    ))
    workflow.add_node("select_evidence", RunnableLambda(
        evidence_selection_node(client, assessment_model),
        afunc=aevidence_selection_node(client, assessment_model),
    ))
    workflow.add_edge(START, "paraphrase")
    workflow.add_edge("paraphrase", "scope")
    workflow.add_conditional_edges("scope", route_scope, {
        "reject": "reject", "retrieve": "retrieve", "fallback": END,
    })
    workflow.add_edge("reject", END)
    workflow.add_conditional_edges("retrieve", route_retrieval, {
        "evidence": "select_evidence", "fallback": END,
    })
    workflow.add_edge("select_evidence", END)
    return workflow.compile()
