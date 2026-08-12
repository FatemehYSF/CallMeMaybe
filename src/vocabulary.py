import json
from pathlib import Path


def load_vocabulary(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_token_to_id(
    vocabulary: dict[str, int],
) -> dict[int, str]:
    return {
        token_id: token
        for token, token_id in vocabulary.items()
    }
