import json
from pathlib import Path
from typing import Any, cast


def load_vocabulary(
    path: Path,
) -> dict[str, Any]:
    """Read the model vocabulary file."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return cast(dict[str, Any], json.load(file))


def build_token_to_id(
    vocabulary: dict[str, Any],
) -> dict[str, int]:
    """Map each token text to its ID."""
    return {
        token: int(token_id)
        for token_id, token in vocabulary.items()
    }
