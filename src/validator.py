from typing import Any

from src.models import FunctionDefinition


def validate_function_call(
    result: dict[str, Any],
    function: FunctionDefinition,
) -> None:
    """Check that a function call has the right shape."""
    # Check the top-level keys
    if set(result) != {
        "prompt",
        "name",
        "parameters",
    }:
        raise ValueError(
            "Invalid function call keys."
        )

    if not isinstance(result["prompt"], str):
        raise ValueError(
            "The prompt must be a string."
        )

    if result["name"] != function.name:
        raise ValueError(
            "The function name does not match."
        )

    parameters = result["parameters"]

    if not isinstance(parameters, dict):
        raise ValueError(
            "Parameters must be an object."
        )

    expected = {
        parameter.name: parameter.type
        for parameter in function.parameters
    }

    if set(parameters) != set(expected):
        raise ValueError(
            "Function parameters do not match."
        )

    for name, parameter_type in expected.items():
        value = parameters[name]

        if parameter_type == "number":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"Parameter '{name}' must be a number."
                )

        elif parameter_type == "string":
            if not isinstance(value, str):
                raise ValueError(
                    f"Parameter '{name}' must be a string."
                )

        elif parameter_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Parameter '{name}' must be a boolean."
                )
