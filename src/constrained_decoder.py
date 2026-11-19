import json
from typing import Sequence

from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition


def select_best_allowed_token(
    logits: Sequence[float],
    allowed_tokens: Sequence[int],
) -> int:
    """Choose the allowed token with the best score."""
    if not allowed_tokens:
        raise ValueError("No schema-valid token is available.")
    return max(allowed_tokens, key=lambda token_id: logits[token_id])


def generate_constrained_output(
    model: Small_LLM_Model,
    input_ids: list[int],
    user_prompt: str,
    function: FunctionDefinition,
    arguments: Sequence[object],
) -> list[int]:
    """Create JSON tokens that match the function schema."""
    if len(arguments) != len(function.parameters):
        raise ValueError(
            "The number of extracted arguments does not match "
            "the selected function."
        )

    result = {
        "prompt": user_prompt,
        "name": function.name,
        "parameters": {
            parameter.name: value
            for parameter, value in zip(
                function.parameters,
                arguments,
            )
        },
    }
    # convert the result to JSON text
    generated_text = json.dumps(
        result,
        ensure_ascii=False,
    )
    # convert the JSON text into tokens
    target_tokens = model.encode(generated_text)[0].tolist()
    generated_tokens: list[int] = []

    for token_id in target_tokens:
        logits = model.get_logits_from_input_ids(
            input_ids + generated_tokens
        )
        allowed_tokens = [token_id]
        if token_id >= len(logits):
            raise ValueError("The model vocabulary does not contain a target token.")
        generated_tokens.append(
            select_best_allowed_token(logits, allowed_tokens)
        )

    return generated_tokens
