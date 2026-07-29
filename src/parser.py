import json
from pathlib import Path
from .models import FunctionDefinition, Parameter
from typing import List


def parse_json(file_path: Path):
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_function_models(data: list[dict]) -> list[FunctionDefinition]:
    function_models: List[FunctionDefinition] = []

    for function_dict in data:
        name = function_dict["name"]
        description = function_dict["description"]
        return_type = function_dict["returns"]["type"]

        parameters: List[Parameter] = []

        for key, value in function_dict["parameters"].items():
            parameter = Parameter(
                name=key,
                type=value["type"]
            )
            parameters.append(parameter)

        function_model = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type
        )

        function_models.append(function_model)

    return function_models
