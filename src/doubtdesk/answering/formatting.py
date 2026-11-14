import re


INLINE_CITATION_PATTERN = re.compile(
    r"\(\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"\s*,\s*p(?:age)?\.?\s*(\d+)\s*\)",
    re.IGNORECASE,
)


PDF_PAGE_PATTERN = re.compile(
    r"\(\s*PDF\s+p(?:age)?\.?\s*(\d+)\s*\)", re.IGNORECASE,
)


def normalize_answer_citations(answer: str, sources: list[dict]) -> str:
    indices = {}
    for source in sources:
        doc_id = str(source["doc_id"]).lower()
        for page in source["page_numbers"]:
            key = (doc_id, int(page))
            if key not in indices:
                indices[key] = len(indices) + 1

    def replace(match: re.Match[str]) -> str:
        key = (match.group(1).lower(), int(match.group(2)))
        index = indices.get(key)
        return f"[{index}]" if index is not None else "[unverified source]"

    def replace_page(match: re.Match[str]) -> str:
        page = int(match.group(1))
        matches = [index for (_, number), index in indices.items() if number == page]
        if len(matches) == 1:
            return f"[{matches[0]}]"
        if matches:
            return "[ambiguous source]"
        return "[unverified source]"

    answer = INLINE_CITATION_PATTERN.sub(replace, answer)
    return PDF_PAGE_PATTERN.sub(replace_page, answer)


def format_answer(
    response: dict,
    document_names: dict[str, str],
    *, normalized=False,
) -> str:
    answer = response["answer"].strip()
    if response["status"] == "grounded" and not normalized:
        answer = normalize_answer_citations(answer, response["sources"])

    if response["status"] == "verification_failed":
        return answer

    if response["status"] == "general_knowledge":
        return f"{answer}\n\nNot from document."

    if response["status"] != "grounded":
        raise ValueError("Unsupported answer status")

    citations = []

    for source in response["sources"]:
        doc_id = source["doc_id"]

        if doc_id not in document_names:
            raise ValueError(
                f"Document filename not found: {doc_id}"
            )

        filename = document_names[doc_id]
        page_numbers = sorted(set(source["page_numbers"]))
        pages = ", ".join(str(number) for number in page_numbers)

        citations.append(f"- {filename}: PDF page(s) {pages}")

    return f"{answer}\n\nSources:\n" + "\n".join(citations)
