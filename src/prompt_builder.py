from src.models import FunctionDefinition

# Worked examples that show the model how words map to literal values.
_ARGUMENT_EXAMPLES = [
    (
        "Function: fn_mask_text\n"
        "Description: Replace every regex match in a text.\n"
        "Parameters:\n- text (string)\n- pattern (string)\n"
        "- mask (string)\n\n"
        "Request:\nReplace every digit in 'room 42b' with hashes",
        '{"prompt": "Replace every digit in \'room 42b\' with hashes", '
        '"name": "fn_mask_text", "parameters": {"text": "room 42b", '
        '"pattern": "\\\\d", "mask": "#"}}',
    ),
    (
        "Function: fn_mask_text\n"
        "Description: Replace every regex match in a text.\n"
        "Parameters:\n- text (string)\n- pattern (string)\n"
        "- mask (string)\n\n"
        "Request:\nChange all consonants b, c and d in 'abcde' to dashes",
        '{"prompt": "Change all consonants b, c and d in \'abcde\' to '
        'dashes", "name": "fn_mask_text", "parameters": {"text": "abcde", '
        '"pattern": "[bcdBCD]", "mask": "-"}}',
    ),
    (
        "Function: fn_power\n"
        "Description: Raise a number to a power.\n"
        "Parameters:\n- base (number)\n- exponent (number)\n\n"
        "Request:\nWhat is seven to the power of 3?",
        '{"prompt": "What is seven to the power of 3?", '
        '"name": "fn_power", "parameters": {"base": 7.0, '
        '"exponent": 3.0}}',
    ),
]


def build_prompt(
    function_models: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Build the message sent to the LLM."""
    prompt_text = "<|im_start|>system\n"
    prompt_text += "Choose exactly one function name from the provided list.\n"
    prompt_text += "<|im_end|>\n<|im_start|>user\nAvailable functions:\n\n"

    for function in function_models:
        parameters = ", ".join(
            f"{parameter.name}: {parameter.type}"
            for parameter in function.parameters
        )
        prompt_text += f"{function.name}({parameters})\n"
        prompt_text += (
            f"Description: {function.description}\n\n"
        )

    prompt_text += "Request:\n"
    prompt_text += f"{user_prompt}\n"
    prompt_text += "Reply with only the matching function name.\n"
    prompt_text += "<|im_end|>\n<|im_start|>assistant\n"
    # Skip Qwen3's thinking block so the name is the very next output.
    prompt_text += "<think>\n\n</think>\n\n"

    return prompt_text


def build_arguments_prompt(
    function: FunctionDefinition,
    user_prompt: str,
) -> str:
    """Build the message that asks the LLM to fill in the arguments."""
    prompt_text = "<|im_start|>system\n"
    prompt_text += (
        "You write function calls as JSON. Fill every parameter with the "
        "exact value the request asks for. When the request names a "
        "symbol or a number in words, write the symbol or number itself, "
        "never its name (for example 'underscores' becomes '_', 'plus "
        "signs' becomes '+', 'ten' becomes 10). A regex parameter must "
        "be a valid Python regular expression.\n"
    )
    prompt_text += "<|im_end|>\n"

    for example_request, example_call in _ARGUMENT_EXAMPLES:
        prompt_text += f"<|im_start|>user\n{example_request}<|im_end|>\n"
        prompt_text += f"<|im_start|>assistant\n{example_call}<|im_end|>\n"

    prompt_text += "<|im_start|>user\n"
    prompt_text += f"Function: {function.name}\n"
    prompt_text += f"Description: {function.description}\n"
    prompt_text += "Parameters:\n"

    for parameter in function.parameters:
        prompt_text += f"- {parameter.name} ({parameter.type})\n"

    prompt_text += f"\nRequest:\n{user_prompt}\n"
    prompt_text += "<|im_end|>\n<|im_start|>assistant\n"
    # Skip Qwen3's thinking block and go straight to the JSON.
    prompt_text += "<think>\n\n</think>\n\n"

    return prompt_text
