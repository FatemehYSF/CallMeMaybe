import json
from pathlib import Path
from .models import FunctionDefinition


def parse_json(file_path: Path):
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_functions_models(data: list[dict]) -> list[FunctionDefinition]:
    pass
