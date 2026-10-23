
def format_answer(
    response: dict,
    document_names: dict[str, str],
) -> str:
    answer = response["answer"].strip()

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
