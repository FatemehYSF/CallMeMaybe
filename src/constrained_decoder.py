from typing import Sequence

from llm_sdk import Small_LLM_Model


def select_best_allowed_token(
    logits: Sequence[float],
    allowed_tokens: list[int],
) -> int:
    """Choose the highest-scoring allowed token."""
    return max(
        allowed_tokens,
        key=lambda token_id: logits[token_id],
    )


def get_next_token_toward_target(
    model: Small_LLM_Model,
    current_tokens: list[int],
    target_text: str,
) -> list[int]:
    """
    Return the next token needed to reach target_text.
    """
    target_tokens = model.encode(target_text)[0].tolist()

    # Current tokens must match the beginning of the target.
    if target_tokens[:len(current_tokens)] != current_tokens:
        return []

    # Target is already complete.
    if len(current_tokens) >= len(target_tokens):
        return []

    # Return the next required token.
    return [target_tokens[len(current_tokens)]]


def get_allowed_tokens(
    model: Small_LLM_Model,
    generated_tokens: list[int],
    target: str,
) -> list[int]:
    """
    Return the token allowed at the current generation state.
    """
    return get_next_token_toward_target(
        model,
        generated_tokens,
        target,
    )


def generate_constrained_prefix(
    model: Small_LLM_Model,
    input_ids: list[int],
    target: str,
) -> list[int]:
    """Generate tokens while following the target exactly."""
    generated_tokens: list[int] = []

    target_tokens = model.encode(target)[0].tolist()

    for _ in range(len(target_tokens)):
        allowed_tokens = get_allowed_tokens(
            model,
            generated_tokens,
            target,
        )

        if not allowed_tokens:
            break

        logits = model.get_logits_from_input_ids(
            input_ids + generated_tokens
        )

        best_token = select_best_allowed_token(
            logits,
            allowed_tokens,
        )

        generated_tokens.append(best_token)

    return generated_tokens


def escape_json_string(value: str) -> str:
    """Escape characters that have special meaning in JSON strings."""
    return (
        value
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )