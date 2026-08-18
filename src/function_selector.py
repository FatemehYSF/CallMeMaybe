import math
from typing import Sequence

from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition


def _log_softmax(logits: Sequence[float], token_id: int) -> float:
    """Calculate how likely one token is.
    It turns the model's raw logits into probabilities."""
    maximum = max(logits)
    total = sum(math.exp(value - maximum) for value in logits)
    return logits[token_id] - maximum - math.log(total)


def select_function(
    model: Small_LLM_Model,
    input_ids: list[int],
    functions: list[FunctionDefinition],
) -> str:
    """Use the LLM to choose the best function."""
    scores: dict[str, float] = {}

    for function in functions:
        name_tokens = model.encode(function.name)[0].tolist()
        current_ids = input_ids.copy()
        log_probability = 0.0

        for token_id in name_tokens:
            logits = model.get_logits_from_input_ids(current_ids)
            log_probability += _log_softmax(logits, token_id)
            current_ids.append(token_id)

        scores[function.name] = log_probability
        print(f"{function.name} → {scores[function.name]}")

    return max(scores, key=lambda name: scores[name])
