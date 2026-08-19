import re
from typing import Any, cast

from src.models import FunctionDefinition


def extract_arguments(
    user_prompt: str,
    function: FunctionDefinition,
) -> list[Any]:
    """Find values in the prompt for each parameter."""
    numbers = [
        float(number)
        for number in re.findall(
            r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])",
            user_prompt,
        )
    ]
    strings = _quoted_strings(user_prompt)
    arguments: list[Any] = []

    for parameter in function.parameters:
        if parameter.type == "number":
            arguments.append(_take_number(numbers, parameter.name))
        elif parameter.type == "boolean":
            arguments.append(_take_boolean(user_prompt, parameter.name))
        else:
            arguments.append(
                _take_string(
                    user_prompt,
                    parameter.name,
                    strings,
                )
            )

    return arguments


def _quoted_strings(user_prompt: str) -> list[str]:
    """Find text inside quotation marks."""
    matches = re.findall(r'"([^\"]*)"|\'([^\']*)\'', user_prompt)
    return [double or single for double, single in matches]


def _take_number(numbers: list[float], parameter_name: str) -> float:
    """Return the next number from the prompt."""
    if not numbers:
        raise ValueError(
            f"Could not find a number for parameter '{parameter_name}'."
        )
    return numbers.pop(0)


def _take_boolean(user_prompt: str, parameter_name: str) -> bool:
    """Return true or false from the prompt."""
    match = re.search(r"\b(true|false)\b", user_prompt, re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Could not find true or false for parameter '{parameter_name}'."
        )
    return cast(str, match.group(1)).lower() == "true"


def _take_string(
    user_prompt: str,
    parameter_name: str,
    strings: list[str],
) -> str:
    """Return one text value from the prompt."""
    if parameter_name.lower() in {"regex", "pattern"}:
        return _regex_value(user_prompt)

    if "source" in parameter_name.lower():
        match = re.search(r"\bin\s+['\"]([^'\"]+)['\"]", user_prompt)
        if match is not None:
            source = match.group(1)
            if source in strings:
                strings.remove(source)
            return source

    if strings:
        return strings.pop(0)

    words = re.findall(r"[\w-]+", user_prompt)
    if not words:
        raise ValueError(
            f"Could not find a string for parameter '{parameter_name}'."
        )
    return cast(str, words[-1])


def _regex_value(user_prompt: str) -> str:
    """Create a regex pattern from the prompt."""
    lower_prompt = user_prompt.lower()
    if "number" in lower_prompt or "digit" in lower_prompt:
        return r"\d+"
    if "vowel" in lower_prompt:
        return r"[aeiou]"

    match = re.search(r"(?:word|pattern)\s+['\"]([^'\"]+)['\"]", user_prompt)
    if match is not None:
        return cast(str, match.group(1))
    raise ValueError("Could not infer a regex or pattern from the prompt.")
