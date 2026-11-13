import json
import re


class IncrementalEvidenceAnswer:
    def __init__(self):
        self.raw = ""
        self.sent = 0

    def feed(self, delta: str) -> str:
        self.raw += delta
        match = re.search(r'"status"\s*:\s*"(sufficient|insufficient)"', self.raw)
        if match is None or match.group(1) != "sufficient":
            return ""
        match = re.search(r'"answer"\s*:\s*"', self.raw)
        if match is None:
            return ""
        start = match.end()
        escaped = False
        end = len(self.raw)
        for index in range(start, len(self.raw)):
            char = self.raw[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                end = index
                break
        fragment = self.raw[start:end]
        if fragment.endswith("\\"):
            fragment = fragment[:-1]
        try:
            decoded = json.loads('"' + fragment + '"')
        except (ValueError, json.JSONDecodeError):
            return ""
        addition = decoded[self.sent:]
        self.sent = len(decoded)
        return addition
