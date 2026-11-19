import json
import re
from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model

from src.function_selector import sequence_log_probability
from src.models import FunctionDefinition, Parameter

MAX_STRING_CHARS = 200
MAX_NUMBER_CHARS = 24

# A JSON string body that may stop in the middle of an escape sequence.
_STRING_BODY_PREFIX = re.compile(
    r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*'
    r'(?:\\(?:u[0-9a-fA-F]{0,3})?)?'
)
_STRING_BODY = re.compile(
    r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*'
)
_NUMBER_PREFIX = {
    "number": re.compile(r"-?(?:\d+(?:\.\d*)?)?"),
    "integer": re.compile(r"-?\d*"),
}
_NUMBER = {
    "number": re.compile(r"-?\d+(?:\.\d+)?"),
    "integer": re.compile(r"-?\d+"),
}


class TokenVocabulary:
    """Group the model tokens by what they can do inside JSON."""

    def __init__(self, model: Small_LLM_Model, size: int) -> None:
        self.texts: list[str] = []
        plain: list[int] = []
        escapes: list[int] = []
        quotes: list[int] = []
        numbers: list[int] = []
        number_ends: list[int] = []

        for token_id in range(size):
            text = model.decode([token_id])
            self.texts.append(text)
            if not text or "�" in text:
                continue
            if re.fullmatch(r"[-0-9.]+", text):
                numbers.append(token_id)
            if text.lstrip()[:1] in {",", "}"}:
                number_ends.append(token_id)
            if '"' in text:
                quotes.append(token_id)
            elif "\\" in text:
                escapes.append(token_id)
            elif _STRING_BODY.fullmatch(text):
                plain.append(token_id)

        self.plain = np.array(plain, dtype=np.int64)
        self.escapes = escapes
        self.quotes = quotes
        self.numbers = numbers
        self.number_ends = np.array(number_ends, dtype=np.int64)


def _best(logits: np.ndarray, token_ids: list[int]) -> tuple[int, float]:
    """Return the allowed token with the highest logit."""
    if not token_ids:
        return -1, float("-inf")
    candidates = np.array(token_ids, dtype=np.int64)
    index = int(np.argmax(logits[candidates]))
    return int(candidates[index]), float(logits[candidates[index]])


def _generate_string(
    model: Small_LLM_Model,
    vocabulary: TokenVocabulary,
    input_ids: list[int],
) -> str:
    """Let the model write a JSON string body until it closes the quote."""
    body = ""
    generated: list[int] = []

    while len(body) < MAX_STRING_CHARS:
        logits = np.array(
            model.get_logits_from_input_ids(input_ids + generated)
        )
        complete = _STRING_BODY.fullmatch(body) is not None

        best_id, best_score = -1, float("-inf")
        if complete and len(vocabulary.plain) > 0:
            index = int(np.argmax(logits[vocabulary.plain]))
            best_id = int(vocabulary.plain[index])
            best_score = float(logits[best_id])

        others = [
            token_id
            for token_id in vocabulary.escapes
            if _STRING_BODY_PREFIX.fullmatch(
                body + vocabulary.texts[token_id]
            )
        ]
        if not complete:
            # Inside an unfinished escape such as "\u00", check every token.
            others += [
                int(token_id)
                for token_id in vocabulary.plain
                if _STRING_BODY_PREFIX.fullmatch(
                    body + vocabulary.texts[token_id]
                )
            ]
        other_id, other_score = _best(logits, others)
        if other_score > best_score:
            best_id, best_score = other_id, other_score

        # Tokens that hold the closing quote, such as '"' or '+"'.
        closing = [
            token_id
            for token_id in vocabulary.quotes
            if _closes_string(body, vocabulary.texts[token_id])
        ]
        close_id, close_score = _best(logits, closing)

        if close_score >= best_score:
            text = vocabulary.texts[close_id]
            return body + text[:text.index('"')]
        if best_id < 0:
            raise ValueError("No schema-valid token is available.")

        generated.append(best_id)
        body += vocabulary.texts[best_id]

    if _STRING_BODY.fullmatch(body) is None:
        raise ValueError("The generated string was cut inside an escape.")
    return body


def _closes_string(body: str, text: str) -> bool:
    """Check that the first quote in a token ends the string."""
    before_quote = body + text[:text.index('"')]
    return _STRING_BODY.fullmatch(before_quote) is not None


def _generate_number(
    model: Small_LLM_Model,
    vocabulary: TokenVocabulary,
    input_ids: list[int],
    parameter_type: str,
) -> str:
    """Let the model write a number one token at a time."""
    prefix = _NUMBER_PREFIX[parameter_type]
    number = _NUMBER[parameter_type]
    text = ""
    generated: list[int] = []

    while len(text) < MAX_NUMBER_CHARS:
        logits = np.array(
            model.get_logits_from_input_ids(input_ids + generated)
        )
        allowed = [
            token_id
            for token_id in vocabulary.numbers
            if prefix.fullmatch(text + vocabulary.texts[token_id])
        ]
        best_id, best_score = _best(logits, allowed)

        if number.fullmatch(text):
            stop_score = float(np.max(logits[vocabulary.number_ends]))
            if stop_score >= best_score:
                return text
        if best_id < 0:
            break

        generated.append(best_id)
        text += vocabulary.texts[best_id]

    if number.fullmatch(text) is None:
        raise ValueError("The model did not produce a valid number.")
    return text


def _generate_boolean(
    model: Small_LLM_Model,
    input_ids: list[int],
) -> str:
    """Let the model choose between true and false."""
    return max(
        ("true", "false"),
        key=lambda word: sequence_log_probability(
            model,
            input_ids,
            model.encode(word)[0].tolist(),
        ),
    )


def _generate_value(
    model: Small_LLM_Model,
    vocabulary: TokenVocabulary,
    input_ids: list[int],
    parameter: Parameter,
) -> tuple[Any, str]:
    """Generate one parameter and return its value and JSON text."""
    if parameter.type == "string":
        body = _generate_string(model, vocabulary, input_ids)
        return json.loads(f'"{body}"'), f'{body}"'
    if parameter.type == "boolean":
        word = _generate_boolean(model, input_ids)
        return word == "true", word

    text = _generate_number(model, vocabulary, input_ids, parameter.type)
    if parameter.type == "integer":
        return int(text), text
    return float(text), text


def generate_constrained_output(
    model: Small_LLM_Model,
    vocabulary: TokenVocabulary,
    prompt_ids: list[int],
    user_prompt: str,
    function: FunctionDefinition,
) -> dict[str, Any]:
    """Create a function call whose values are written by the model.

    The JSON structure is forced, and only tokens that keep each value
    valid for its schema type are allowed while the model writes it.
    """
    output_text = (
        f'{{"prompt": {json.dumps(user_prompt, ensure_ascii=False)}, '
        f'"name": {json.dumps(function.name)}, "parameters": {{'
    )
    parameters: dict[str, Any] = {}

    for index, parameter in enumerate(function.parameters):
        if index > 0:
            output_text += ", "
        output_text += f"{json.dumps(parameter.name)}: "
        if parameter.type == "string":
            output_text += '"'

        input_ids = prompt_ids + model.encode(output_text)[0].tolist()
        value, value_text = _generate_value(
            model,
            vocabulary,
            input_ids,
            parameter,
        )
        parameters[parameter.name] = value
        output_text += value_text

    return {
        "prompt": user_prompt,
        "name": function.name,
        "parameters": parameters,
    }
