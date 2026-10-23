from typing import Literal
from doubtdesk.graph.state import DoubtDeskState


def route_documents(
    state: DoubtDeskState,
) -> Literal["retrieve", "fallback"]:
    if state["doc_ids"]:
        return "retrieve"

    return "fallback"


def route_evidence(
    state: DoubtDeskState,
) -> Literal["grounded", "fallback"]:
    if state["evidence"].status == "sufficient":
        return "grounded"

    return "fallback"
