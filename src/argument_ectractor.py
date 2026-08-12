import re

from src.models import FunctionDefinition


def extract_arguments(
    user_prompt: str,
    function: FunctionDefinition,
) -> list[object]:
    numbers = re.findall(r"\d+(?:\.\d+)?", user_prompt)
    strings = re.findall(r"'([^']*)'|\"([^\"]*)\"", user_prompt)

    extracted_strings = [
        single or double
        for single, double in strings
    ]

    arguments: list[object] = []

    for parameter in function.parameters:
        if parameter.type == "number":
            if not numbers:
                raise ValueError(
                    f"Could not find a number for parameter "
                    f"'{parameter.name}'."
                )

            arguments.append(float(numbers.pop(0)))

        elif parameter.type == "string":
            if extracted_strings:
                arguments.append(extracted_strings.pop(0))
            else:
                words = user_prompt.split()
                arguments.append(words[-1])

    return arguments
