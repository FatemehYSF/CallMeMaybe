import json
from pathlib import Path
from typing import Any

from src.models import FunctionDefinition, Parameter


def parse_json(path: Path) -> list[dict[str, Any]]:
    """Read a JSON file that contains a list."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {path}"
        )

    return data


def build_function_models(
    functions: list[dict[str, Any]],
) -> list[FunctionDefinition]:
    """Turn JSON function data into validated models."""
    models: list[FunctionDefinition] = []

    for function in functions:
        parameters = [
            Parameter(
                name=name,
                type=details["type"],
            )
            for name, details in function["parameters"].items()
        ]

        models.append(
            FunctionDefinition(
                name=function["name"],
                description=function["description"],
                parameters=parameters,
                return_type=function["returns"]["type"],
            )
        )

    return models
