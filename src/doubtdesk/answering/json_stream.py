import re

_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
CITATION_PATTERN = re.compile(
    r"\(\s*(?:(?P<alias>S\d+)"
    r"|PDF\s+p(?:age)?\.?\s*(?P<pdf_page>\d+)"
    r"|(?P<doc>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\s*,\s*p(?:age)?\.?\s*(?P<doc_page>\d+))"
    r"\s*\)",
    re.IGNORECASE,
)
HOLD_BACK_CHARS = 64


def resolve_citation(match: re.Match, aliases: dict):
    if match.group("alias"):
        return aliases.get(match.group("alias").upper())
    if match.group("pdf_page"):
        page = int(match.group("pdf_page"))
        keys = {key for key in aliases.values() if key[1] == page}
        return next(iter(keys)) if len(keys) == 1 else None
    key = (match.group("doc").lower(), int(match.group("doc_page")))
    return next((value for value in aliases.values() if (str(value[0]).lower(), value[1]) == key), None)


def citation_keys_in_order(text: str, aliases: dict) -> list:
    aliases = {alias.upper(): key for alias, key in aliases.items()}
    keys = (resolve_citation(match, aliases) for match in CITATION_PATTERN.finditer(text))
    return list(dict.fromkeys(key for key in keys if key is not None))


class AnswerFieldStreamer:
    def __init__(self, field: str = "answer", status_field: str = "status"):
        self.field = field
        self.status_field = status_field
        self.status: str | None = None
        self.answer_started = False
        self.answer_complete = False
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._unicode: str | None = None
        self._high_surrogate: int | None = None
        self._expect_key = False
        self._last_key: str | None = None
        self._value_for: str | None = None
        self._target: str | None = None
        self._buffer: list[str] = []

    def feed(self, delta: str) -> str:
        output: list[str] = []
        for char in delta:
            if self._in_string:
                self._string_char(char, output)
            else:
                self._structural_char(char)
        return "".join(output)

    def _emit(self, text: str, output: list[str]):
        if self._target == "answer":
            output.append(text)
        elif self._target in {"key", "status"}:
            self._buffer.append(text)

    def _emit_codepoint(self, code: int, output: list[str]):
        if 0xD800 <= code <= 0xDBFF:
            if self._high_surrogate is not None:
                self._emit("�", output)
            self._high_surrogate = code
            return
        if 0xDC00 <= code <= 0xDFFF:
            if self._high_surrogate is None:
                self._emit("�", output)
                return
            combined = 0x10000 + ((self._high_surrogate - 0xD800) << 10) + (code - 0xDC00)
            self._high_surrogate = None
            self._emit(chr(combined), output)
            return
        self._flush_surrogate(output)
        self._emit(chr(code), output)

    def _flush_surrogate(self, output: list[str]):
        if self._high_surrogate is not None:
            self._high_surrogate = None
            self._emit("�", output)

    def _string_char(self, char: str, output: list[str]):
        if self._unicode is not None:
            self._unicode += char
            if len(self._unicode) == 4:
                try:
                    code = int(self._unicode, 16)
                except ValueError:
                    code = 0xFFFD
                self._unicode = None
                self._emit_codepoint(code, output)
            return
        if self._escape:
            self._escape = False
            if char == "u":
                self._unicode = ""
                return
            self._flush_surrogate(output)
            self._emit(_SIMPLE_ESCAPES.get(char, char), output)
            return
        if char == "\\":
            self._escape = True
            return
        if char == '"':
            self._flush_surrogate(output)
            self._end_string()
            return
        self._flush_surrogate(output)
        self._emit(char, output)

    def _end_string(self):
        self._in_string = False
        value = "".join(self._buffer)
        self._buffer = []
        if self._target == "key":
            self._last_key = value
        elif self._target == "status":
            self.status = value
        elif self._target == "answer":
            self.answer_complete = True
        self._target = None

    def _structural_char(self, char: str):
        if char in "{[":
            self._depth += 1
            if char == "{" and self._depth == 1:
                self._expect_key = True
            return
        if char in "}]":
            self._depth -= 1
            return
        if self._depth != 1:
            if char == '"':
                self._in_string = True
                self._target = None
            return
        if char == ":":
            self._expect_key = False
            self._value_for = self._last_key
            return
        if char == ",":
            self._expect_key = True
            self._value_for = None
            return
        if char == '"':
            self._in_string = True
            if self._expect_key:
                self._target = "key"
            elif self._value_for == self.field:
                self._target = "answer"
                self.answer_started = True
            elif self._value_for == self.status_field:
                self._target = "status"
            else:
                self._target = None


class AliasRewriter:
    def __init__(self, aliases: dict):
        self.aliases = {alias.upper(): key for alias, key in aliases.items()}
        self._indices: dict = {}
        self._pending = ""

    def _replace(self, match: re.Match) -> str:
        key = resolve_citation(match, self.aliases)
        if key is None:
            return match.group(0)
        if key not in self._indices:
            self._indices[key] = len(self._indices) + 1
        return f"[{self._indices[key]}]"

    def feed(self, text: str) -> str:
        data = self._pending + text
        self._pending = ""
        cut = data.rfind("(")
        if cut != -1:
            tail = data[cut:]
            if ")" not in tail and len(tail) <= HOLD_BACK_CHARS:
                self._pending = tail
                data = data[:cut]
        return CITATION_PATTERN.sub(self._replace, data)

    def flush(self) -> str:
        data, self._pending = self._pending, ""
        return CITATION_PATTERN.sub(self._replace, data)
