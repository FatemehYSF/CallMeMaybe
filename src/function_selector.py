import math

from llm_sdk import Small_LLM_Model


def log_softmax_value(
    logits: list[float],
    token_id: int,
) -> float:
    max_logit = max(logits)

    total = sum(
        math.exp(logit - max_logit)
        for logit in logits
    )

    return logits[token_id] - max_logit - math.log(total)


def score_function(
    model: Small_LLM_Model,
    input_ids: list[int],
    function_tokens: list[int],
) -> float:
    score = 0.0
    generated_tokens = [function_tokens[0]]

    for token_id in function_tokens[1:]:
        current_input = input_ids + generated_tokens

        logits = model.get_logits_from_input_ids(
            current_input
        )

        score += log_softmax_value(
            logits,
            token_id,
        )

        generated_tokens.append(token_id)

    return score / len(function_tokens[1:])


def select_function(
    model: Small_LLM_Model,
    input_ids: list[int],
    function_tokens: dict[str, list[int]],
) -> str:
    best_function = ""
    best_score = float("-inf")

    for function_name, tokens in function_tokens.items():
        score = score_function(
            model,
            input_ids,
            tokens,
        )

        print(function_name, "→", score)

        if score > best_score:
            best_score = score
            best_function = function_name

    return best_function
