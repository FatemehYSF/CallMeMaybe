from src.models import FunctionDefinition


def build_prompt(
    function_models: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    prompt_text = "Available functions:\n\n"

    for function in function_models:
        prompt_text += f"Function: {function.name}\n"
        prompt_text += f"Description: {function.description}\n\n"

    prompt_text += "User:\n"
    prompt_text += f"{user_prompt}\n\n"
    prompt_text += "Choose the function that best matches the user's request.\n"
    prompt_text += "Respond with only the function name.\n"
    prompt_text += "Answer:\n"

    return prompt_text
