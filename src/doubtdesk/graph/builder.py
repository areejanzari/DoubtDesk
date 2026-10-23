from pathlib import Path
from langgraph.graph import StateGraph, START, END
from doubtdesk.graph.state import DoubtDeskState
from doubtdesk.graph.nodes import (
    retrieve_node,
    evidence_node,
    fallback_node,
    grounded_node,
    formatting_node,
)
from doubtdesk.graph.routing import (
    route_documents,
    route_evidence,
)


def build_graph(
    collection,
    embedding_model,
    client,
    model_name: str,
    db_path: Path = Path("data/documents.db"),
):
    workflow = StateGraph(DoubtDeskState)

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
    

    workflow.add_conditional_edges(
        START,
        route_documents,
        {
            "retrieve": "retrieve",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("retrieve", "evidence")

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
