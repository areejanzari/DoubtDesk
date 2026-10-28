from pathlib import Path
from langgraph.graph import StateGraph, START, END
from doubtdesk.graph.state import DoubtDeskState
from doubtdesk.graph.nodes import (
    retrieve_node,
    evidence_node,
    fallback_node,
    grounded_node,
    formatting_node,
    paraphrase_node,
    scope_node,
    rejection_node,
)
from doubtdesk.graph.routing import (
    route_documents,
    route_evidence,
    route_retrieval,
)


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
):
    workflow = StateGraph(DoubtDeskState)

    workflow.add_node(
        "paraphrase",
        paraphrase_node(client, model_name),
    )

    workflow.add_node(
        "scope",
        scope_node(scope_classifier),
    )

    workflow.add_node(
        "reject",
        rejection_node,
    )

    workflow.add_node(
        "retrieve",
        retrieve_node(collection, embedding_model, db_path),
    )

    workflow.add_node(
        "evidence",
        evidence_node(client, model_name),
    )

    workflow.add_node(
        "fallback",
        fallback_node(client, model_name),
    )

    workflow.add_node("grounded", grounded_node)

    workflow.add_node(
        "format",
        formatting_node(db_path),
    )

    workflow.add_edge(START, "paraphrase")
    workflow.add_edge("paraphrase", "scope")

    workflow.add_conditional_edges(
        "scope",
        route_scope,
        {
            "reject": "reject",
            "retrieve": "retrieve",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("reject", END)

    workflow.add_conditional_edges(
        "retrieve",
        route_retrieval,
        {
            "evidence": "evidence",
            "fallback": "fallback",
        },
    )
    
    workflow.add_conditional_edges(
        "evidence",
        route_evidence,
        {
            "grounded": "grounded",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("grounded", "format")
    workflow.add_edge("fallback", "format")
    workflow.add_edge("format", END)

    return workflow.compile()