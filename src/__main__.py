from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.parser import parse_json, build_function_models
from src.prompt_builder import build_prompt
from src.function_selector import select_function
from src.argument_ectractor import extract_arguments
from src.vocabulary import load_vocabulary, build_token_to_id
from src.constrained_decoder import (
    generate_constrained_prefix,
    escape_json_string,
)


print("Call Me Maybe started!")

model = Small_LLM_Model()
print("Model loaded successfully!")

vocab_path = Path(model.get_path_to_vocab_file())
vocabulary = load_vocabulary(vocab_path)
token_to_id = build_token_to_id(vocabulary)

print(f"Vocabulary size: {len(vocabulary)}")

for text in ['{', '"prompt"', '"name"', '"parameters"', ':', ',']:
    tokens = model.encode(text)
    print(text, tokens[0].tolist())

functions = parse_json(
    Path("data/input/functions_definition.json")
)

prompts = parse_json(
    Path("data/input/function_calling_tests.json")
)

function_models = build_function_models(functions)

print(f"Loaded {len(functions)} functions.")
print(f"Loaded {len(prompts)} prompts.")

function_tokens: dict[str, list[int]] = {}

for function in function_models:
    tokens = model.encode(function.name)[0].tolist()
    function_tokens[function.name] = tokens

for prompt in prompts:
    user_prompt = prompt["prompt"]

    print(f"user: {user_prompt}")

    full_prompt = build_prompt(
        function_models,
        user_prompt,
    )

    input_ids = model.encode(full_prompt)
    input_ids_list = input_ids[0].tolist()

    selected_function = select_function(
        model,
        input_ids_list,
        function_tokens,
    )

    print(f"Selected function: {selected_function}")

    escaped_prompt = escape_json_string(
        user_prompt
    )
    function = next(
        function
        for function in function_models
        if function.name == selected_function
    )

    arguments = extract_arguments(
        user_prompt,
        function,
    )
    parameter_parts = []

    for parameter, value in zip(
        function.parameters,
        arguments,
    ):
        if parameter.type == "string":
            escaped_value = escape_json_string(str(value))
            parameter_parts.append(
                f'"{parameter.name}":"{escaped_value}"'
            )
        else:
            parameter_parts.append(
                f'"{parameter.name}":{value}'
            )

    parameters_json = ",".join(parameter_parts)
    target = (
        f'{{"prompt":"{escaped_prompt}",'
        f'"name":"{selected_function}",'
        f'"parameters":{{{parameters_json}}}}}'
    )

    generated_tokens = generate_constrained_prefix(
        model,
        input_ids_list,
        target,
    )

    generated_text = model.decode(generated_tokens)

    print("Constrained output:", generated_text)

    print(f"Arguments: {arguments}")
