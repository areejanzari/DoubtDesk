import re
from collections import Counter

import pymupdf

EDGE_LINES = 2
REPEAT_MIN_PAGES = 3
REPEAT_PAGE_FRACTION = 0.3
ANYWHERE_PAGE_FRACTION = 0.5
SHADOW_OVERLAP = 0.7
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?[\divxlc]{1,6}\s*$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"[ \t ]+")


def _overlap(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1.0
    return width * height / smaller


def extract_lines(page, textpage=None) -> list[dict]:
    height = page.rect.height or 1.0
    lines = []
    drawn: dict[str, list] = {}
    flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_IMAGES
    for block_index, block in enumerate(page.get_text("dict", textpage=textpage, flags=flags).get("blocks", [])):
        for line in block.get("lines", []):
            parts = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                key = span_text.strip()
                if key:
                    boxes = drawn.setdefault(key, [])
                    if any(_overlap(span["bbox"], box) > SHADOW_OVERLAP for box in boxes):
                        continue
                    boxes.append(span["bbox"])
                parts.append(span_text)
            text = _SPACES.sub(" ", "".join(parts)).strip()
            if not text:
                continue
            _, y0, _, y1 = line["bbox"]
            dx, dy = line.get("dir", (1.0, 0.0))
            lines.append({
                "text": text,
                "block": block_index,
                "top": y0 / height,
                "bottom": y1 / height,
                "rotated": abs(dy) > 0.2 or dx < 0,
            })
    return lines


def plain_lines(text: str) -> list[dict]:
    lines = []
    for index, line in enumerate(text.splitlines()):
        line = _SPACES.sub(" ", line).strip()
        lines.append({"text": line, "block": index, "top": None, "bottom": None, "rotated": False})
    return lines


def _line_key(text: str) -> str:
    return _DIGITS.sub("#", text.casefold()).strip()


def edge_line_ids(lines: list[dict]) -> set[int]:
    positioned = [index for index, line in enumerate(lines) if line["top"] is not None and line["text"]]
    if not positioned:
        return set()
    ordered = sorted(positioned, key=lambda index: lines[index]["top"])
    top_cut = lines[ordered[min(EDGE_LINES, len(ordered)) - 1]]["bottom"]
    bottom_cut = lines[ordered[-min(EDGE_LINES, len(ordered))]]["top"]
    return {index for index in positioned
            if lines[index]["bottom"] <= top_cut + 1e-6 or lines[index]["top"] >= bottom_cut - 1e-6}


def boilerplate_keys(pages_lines: list[list[dict]]) -> set[str]:
    pages = len(pages_lines)
    edge_threshold = max(REPEAT_MIN_PAGES, REPEAT_PAGE_FRACTION * pages)
    anywhere_threshold = max(REPEAT_MIN_PAGES, ANYWHERE_PAGE_FRACTION * pages)
    edges, anywhere = Counter(), Counter()
    for lines in pages_lines:
        edge_ids = edge_line_ids(lines)
        edges.update({_line_key(lines[index]["text"]) for index in edge_ids})
        anywhere.update({line["text"].casefold() for line in lines if len(line["text"]) >= 12})
    return ({key for key, count in edges.items() if count >= edge_threshold}
            | {key for key, count in anywhere.items() if count >= anywhere_threshold})


def clean_page_text(lines: list[dict], boilerplate: set[str]) -> str:
    edge_ids = edge_line_ids(lines)
    blocks: list[list[str]] = []
    previous_block = None
    previous_text = None
    for index, line in enumerate(lines):
        text = line["text"]
        if not text:
            previous_block = None
            continue
        if line["rotated"] or text.casefold() in boilerplate or (index in edge_ids and _line_key(text) in boilerplate):
            continue
        if index in edge_ids and _PAGE_NUMBER.match(text):
            continue
        if text == previous_text:
            continue
        previous_text = text
        if line["block"] != previous_block or not blocks:
            blocks.append([])
            previous_block = line["block"]
        blocks[-1].append(text)
    return "\n\n".join("\n".join(block) for block in blocks if block)


def clean_document(pages_lines: list[list[dict]]) -> list[str]:
    boilerplate = boilerplate_keys(pages_lines)
    return [clean_page_text(lines, boilerplate) for lines in pages_lines]
