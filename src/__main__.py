from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.parser import parse_json, build_function_models
from src.prompt_builder import build_prompt
from src.function_selector import select_function


print("Call Me Maybe started!")

# Load the model ONCE
model = Small_LLM_Model()
print("Model loaded successfully!")

# Load data
functions = parse_json(
    Path("data/input/functions_definition.json")
)
prompts = parse_json(
    Path("data/input/function_calling_tests.json")
)

function_models = build_function_models(functions)

# Encode function names once
function_tokens: dict[str, list[int]] = {}

for function in function_models:
    tokens = model.encode(function.name)[0].tolist()
    function_tokens[function.name] = tokens

print(f"Loaded {len(functions)} functions.")
print(f"Loaded {len(prompts)} prompts.")

# Process prompts
for prompt in prompts:
    print(f"user: {prompt['prompt']}")

    full_prompt = build_prompt(
        function_models,
        prompt["prompt"],
    )

    input_ids = model.encode(full_prompt)
    input_ids_list = input_ids[0].tolist()

    selected_function = select_function(
        model,
        input_ids_list,
        function_tokens,
    )

    print(f"Selected function: {selected_function}")
