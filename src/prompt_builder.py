from src.models import FunctionDefinition


def build_prompt(
    function_models: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Build the message sent to the LLM."""
    prompt_text = "<|im_start|>system\n"
    prompt_text += "Choose exactly one function name from the provided list.\n"
    prompt_text += "<|im_end|>\n<|im_start|>user\nAvailable functions:\n\n"

    for function in function_models:
        prompt_text += f"{function.name}\n"
        prompt_text += (
            f"Description: {function.description}\n\n"
        )

    prompt_text += "Request:\n"
    prompt_text += f"{user_prompt}\n"
    prompt_text += "Reply with only the matching function name.\n"
    prompt_text += "<|im_end|>\n<|im_start|>assistant\n"

    return prompt_text
