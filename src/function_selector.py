import torch
from llm_sdk import Small_LLM_Model


def score_function(
    model: Small_LLM_Model,
    input_ids: list[int],
    function_tokens: list[int],
) -> float:
    score = 0.0
    generated_tokens = [function_tokens[0]]

    for token_id in function_tokens[1:]:
        current_input = input_ids + generated_tokens

        logits = model.get_logits_from_input_ids(current_input)

        logits_tensor = torch.tensor(logits)
        log_probs = torch.log_softmax(logits_tensor, dim=0)

        score += float(log_probs[token_id])

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
