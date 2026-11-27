import re
from doubtdesk.ingestion.schemas import PageRecord, ChunkRecord

CHUNKER_VERSION = 2
TARGET_TOKENS = 200
MAX_TOKENS = 240
OVERLAP_MAX_TOKENS = 60
SPECIAL_TOKENS = 2
HEADING_MAX_CHARS = 90
STRUCTURED_SYMBOL_RATIO = 0.3

_ABBREVIATIONS = frozenset(
    "fig figs eq eqs no nos vs etc approx e.g i.e cf al dr mr mrs ms prof st sec ch vol pp p viz resp max min ref".split()
)
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)+\.?\s+[A-Z(]|(?:chapter|unit|section|part)\s+\w+)[^\n]{0,85}$")
_SECTION_NUMBER = re.compile(r"^\d+(?:\.\d+)+\.?$")
_BULLET = re.compile(r"^(?:[•▪◦●\-–*]|\(?[ivx]{1,4}\)|\(?[a-z]\)|\d{1,2}[.)])\s+", re.IGNORECASE)
_BOUNDARY = re.compile(r"[.!?…]['\"”’)\]]*\s+")
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\n([a-z])")
_SENTENCE_END = re.compile(r"[.!?:…]['\"”’)\]]*$")
MERGE_TAIL_TOKENS = 40
_CLAUSE = re.compile(r"(?<=[;:,])\s+")


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def counting_tokenizer(tokenizer):
    from tokenizers import Tokenizer
    copy = Tokenizer.from_str(tokenizer.to_str())
    copy.no_truncation()
    copy.no_padding()
    return copy


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > HEADING_MAX_CHARS or line.endswith((".", ",", ";")):
        return False
    if _HEADING.match(line):
        return True
    letters = [char for char in line if char.isalpha()]
    return len(letters) >= 6 and len(line.split()) >= 2 and sum(char.isupper() for char in letters) / len(letters) > 0.85


def _is_structured(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    symbols = sum(1 for char in stripped if not (char.isalnum() or char.isspace()))
    words = stripped.split()
    short = len(words) <= 3 and len(stripped) <= 24
    return symbols / len(stripped) >= STRUCTURED_SYMBOL_RATIO or (short and any(char.isdigit() for char in stripped)) or "\t" in line


def split_sentences(text: str) -> list[str]:
    sentences = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        before = text[start:match.start() + 1].rstrip()
        last_word = before.split()[-1] if before.split() else ""
        token = last_word.rstrip(".!?…'\"”’)]").casefold()
        following = text[end:end + 1]
        if token in _ABBREVIATIONS or token.replace(".", "") in _ABBREVIATIONS:
            continue
        if len(token) == 1 and token.isalpha() and re.match(r"[A-Z]\.", text[end:end + 2]):
            continue
        if following and (following.isdigit() or following.islower()):
            continue
        sentences.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [sentence for sentence in sentences if sentence]


def _join_lines(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return re.sub(r"\s*\n\s*", " ", text).strip()


def page_units(text: str) -> list[tuple[str, str | None]]:
    units: list[tuple[str, str | None]] = []
    heading = None
    pending_number = None
    for block in re.split(r"\n\s*\n", text or ""):
        flowing: list[str] = []

        def flush():
            if flowing:
                for sentence in split_sentences(_join_lines(flowing)):
                    units.append((sentence, heading))
                flowing.clear()

        for raw in block.split("\n"):
            line = raw.strip()
            if not line:
                continue
            mid_sentence = bool(flowing) and not _SENTENCE_END.search(flowing[-1])
            if _SECTION_NUMBER.match(line) and not mid_sentence:
                flush()
                pending_number = line
                continue
            if pending_number is not None:
                candidate = f"{pending_number} {line}"
                pending_number = None
                if _is_heading(candidate):
                    flush()
                    heading = candidate
                    continue
                line = candidate
            if not mid_sentence and _is_heading(line):
                flush()
                heading = line
                continue
            if _BULLET.match(line) and flowing:
                flush()
            if _is_structured(line):
                flush()
                if units and units[-1][1] == heading and units[-1][0].endswith("⁣"):
                    units[-1] = (units[-1][0][:-1] + "\n" + line + "⁣", heading)
                else:
                    units.append((line + "⁣", heading))
                continue
            flowing.append(line)
        flush()
    if pending_number is not None:
        units.append((pending_number, heading))
    return [(unit.replace("⁣", ""), heading) for unit, heading in units]


def _split_oversized(tokenizer, text: str, limit: int) -> list[str]:
    pieces = [piece for piece in _CLAUSE.split(text) if piece]
    if len(pieces) > 1 and all(count_tokens(tokenizer, piece) <= limit for piece in pieces):
        return pieces
    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = encoding.offsets
    result = []
    start_token = 0
    while start_token < len(offsets):
        end_token = min(start_token + limit, len(offsets))
        if end_token < len(offsets):
            cut = end_token
            while cut > start_token + limit // 2 and offsets[cut][0] == offsets[cut - 1][1]:
                cut -= 1
            end_token = cut if cut > start_token + limit // 2 else end_token
        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]
        result.append(text[start_char:end_char].strip())
        start_token = end_token
    return [piece for piece in result if piece]


def chunk_page_text(text: str, tokenizer, target_tokens: int = TARGET_TOKENS, max_tokens: int = MAX_TOKENS,
                    overlap_sentences: int = 1) -> list[str]:
    if not 0 < target_tokens <= max_tokens:
        raise ValueError("target_tokens must be positive and no larger than max_tokens")
    tokenizer = counting_tokenizer(tokenizer)
    budget = max_tokens - SPECIAL_TOKENS
    units = []
    for unit, heading in page_units(text):
        prefix = f"Section: {heading}\n" if heading else ""
        prefix_tokens = count_tokens(tokenizer, prefix) if prefix else 0
        limit = max(16, budget - prefix_tokens)
        tokens = count_tokens(tokenizer, unit)
        if tokens > limit:
            for piece in _split_oversized(tokenizer, unit, limit):
                units.append((piece, heading, count_tokens(tokenizer, piece)))
        else:
            units.append((unit, heading, tokens))

    chunks: list[str] = []
    current: list[tuple[str, str | None, int]] = []

    def size(items):
        if not items:
            return 0
        heading = items[0][1]
        prefix_tokens = count_tokens(tokenizer, f"Section: {heading}\n") if heading else 0
        return prefix_tokens + sum(item[2] for item in items)

    def render(items):
        heading = items[0][1]
        body = " ".join(item[0] for item in items)
        return f"Section: {heading}\n{body}" if heading else body

    groups: list[list[tuple[str, str | None, int]]] = []
    for unit in units:
        heading_changed = bool(current) and unit[1] != current[0][1]
        if current and (heading_changed or size(current + [unit]) > target_tokens - SPECIAL_TOKENS):
            groups.append(current)
            carry = [] if heading_changed else current[-overlap_sentences:] if overlap_sentences else []
            carry = [item for item in carry if item[2] <= OVERLAP_MAX_TOKENS]
            current = carry if size(carry + [unit]) <= budget else []
        current.append(unit)
    if current:
        groups.append(current)
    merged: list[list[tuple[str, str | None, int]]] = []
    for group in groups:
        if merged and group[0][1] == merged[-1][0][1]:
            previous_ids = {id(item) for item in merged[-1]}
            fresh = [item for item in group if id(item) not in previous_ids]
            if fresh and size(fresh) < MERGE_TAIL_TOKENS and size(merged[-1] + fresh) <= budget:
                merged[-1] = merged[-1] + fresh
                continue
        merged.append(group)
    chunks = [render(group) for group in merged]
    return [chunk for chunk in chunks if chunk.strip()]


def chunk_document(
    pages: list[PageRecord],
    tokenizer,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_sentences: int = 1,
) -> list[ChunkRecord]:
    all_chunks = []
    for page in pages:
        for index, text in enumerate(chunk_page_text(page.text, tokenizer, target_tokens, max_tokens, overlap_sentences)):
            all_chunks.append(ChunkRecord(
                doc_id=page.doc_id,
                page_number=page.page_number,
                chunk_index=index,
                text=text,
            ))
    return all_chunks
