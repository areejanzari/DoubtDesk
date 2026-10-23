from doubtdesk.answering.schemas import EvidenceResult


def validate_evidence_sources(
    result: EvidenceResult,
    pages: list[dict],
) -> None:
    allowed_pages = {
        (page["doc_id"], page["page_number"])
        for page in pages
    }

    for source in result.sources:
        for page_number in source.page_numbers:
            key = (source.doc_id, page_number)

            if key not in allowed_pages:
                raise ValueError(
                    f"Invalid citation: {source.doc_id}, "
                    f"page {page_number}"
                )


def validate_evidence_consistency(
    result: EvidenceResult,
) -> None:
    if result.status == "sufficient":
        if not result.answer.strip():
            raise ValueError(
                "Sufficient evidence requires an answer"
            )

        if not result.sources:
            raise ValueError(
                "Sufficient evidence requires citations"
            )

        if not any(
            source.page_numbers for source in result.sources
        ):
            raise ValueError(
                "Sufficient evidence requires page numbers"
            )

    else:
        if result.answer.strip() or result.sources:
            raise ValueError(
                "Insufficient evidence must not contain "
                "an answer or citations"
            )
