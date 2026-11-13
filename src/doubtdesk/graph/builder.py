from pathlib import Path
from langgraph.graph import StateGraph, START, END
from doubtdesk.graph.state import DoubtDeskState
from doubtdesk.graph.nodes import (
    retrieve_node,
    evidence_selection_node,
    evidence_node,
    fallback_node,
    grounded_node,
    formatting_node,
    paraphrase_node,
    scope_node,
    rejection_node,
    insufficient_document_node,
    route_selected_evidence,
    output_guardrail_node,
    route_after_output_guardrail,
)
from doubtdesk.graph.routing import route_documents, route_evidence, route_retrieval


def route_scope(state: DoubtDeskState) -> str:
    if state["is_off_topic"]:
        return "reject"
    return route_documents(state)


def build_graph(
    collection,
    embedding_model,
    client,
    model_name: str,
    scope_classifier,
    db_path: Path = Path("data/documents.db"),
    memory_collection=None,
    output_judge=None,
    paraphrase_model_name=None,
    assessment_model_name=None,
):
    workflow = StateGraph(DoubtDeskState)
    workflow.add_node("paraphrase", paraphrase_node(client, paraphrase_model_name or model_name, embedding_model, memory_collection))
    workflow.add_node("scope", scope_node(scope_classifier))
    workflow.add_node("reject", rejection_node)
    workflow.add_node("retrieve", retrieve_node(collection, embedding_model, db_path))
    workflow.add_node("select_evidence", evidence_selection_node(client, assessment_model_name or model_name))
    workflow.add_node("evidence", evidence_node(client, model_name))
    workflow.add_node("fallback", fallback_node(client, model_name))
    workflow.add_node("grounded", grounded_node)
    workflow.add_node("insufficient_document", insufficient_document_node)
    if output_judge is not None:
        workflow.add_node("output_guardrail", output_guardrail_node(client, model_name, output_judge))
    workflow.add_node("format", formatting_node(db_path))
    workflow.add_edge(START, "paraphrase")
    workflow.add_edge("paraphrase", "scope")
    workflow.add_conditional_edges("scope", route_scope, {
        "reject": "reject", "retrieve": "retrieve", "fallback": "fallback",
    })
    workflow.add_edge("reject", END)
    workflow.add_conditional_edges("retrieve", route_retrieval, {
        "evidence": "select_evidence", "fallback": "fallback",
    })
    workflow.add_conditional_edges("select_evidence", route_selected_evidence, {
        "evidence": "evidence",
        "fallback": "fallback",
        "insufficient_document": "insufficient_document",
    })
    workflow.add_conditional_edges("evidence", route_evidence, {
        "grounded": "grounded", "fallback": "fallback",
    })
    if output_judge is None:
        workflow.add_edge("grounded", "format")
        workflow.add_edge("fallback", "format")
    else:
        workflow.add_edge("grounded", "output_guardrail")
        workflow.add_edge("fallback", "output_guardrail")
        workflow.add_conditional_edges("output_guardrail", route_after_output_guardrail, {
            "fallback": "fallback", "format": "format",
        })
    workflow.add_edge("insufficient_document", "format")
    workflow.add_edge("format", END)
    return workflow.compile()


def build_preparation_graph(
    collection, embedding_model, client, model_name, scope_classifier,
    db_path=Path("data/documents.db"), memory_collection=None,
    paraphrase_model_name=None, assessment_model_name=None,
):
    workflow = StateGraph(DoubtDeskState)
    workflow.add_node("paraphrase", paraphrase_node(client, paraphrase_model_name or model_name, embedding_model, memory_collection))
    workflow.add_node("scope", scope_node(scope_classifier))
    workflow.add_node("reject", rejection_node)
    workflow.add_node("retrieve", retrieve_node(collection, embedding_model, db_path))
    workflow.add_node("select_evidence", evidence_selection_node(client, assessment_model_name or model_name))
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
