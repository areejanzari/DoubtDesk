import hashlib
from pathlib import Path

def calculate_sha256(file_path: Path) -> str:
    with file_path.open("rb") as file:
        digest = hashlib.file_digest(file,"sha256")
    return digest.hexdigest()