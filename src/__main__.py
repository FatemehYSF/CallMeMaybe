from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.parser import parse_json, build_function_models

print("Call Me Maybe started!")

# Load the model ONCE
model = Small_LLM_Model()
print("Model loaded successfully!")

# Load data
functions = parse_json(Path("data/input/functions_definition.json"))
prompts = parse_json(Path("data/input/function_calling_tests.json"))

function_models = build_function_models(functions)

print(f"Loaded {len(functions)} functions.")
print(f"Loaded {len(prompts)} promts.")
# for function in function_models:
#    print(function)

for prompt in prompts:
    print(f"user: {prompt['prompt']}")

    input_ids = model.encode(prompt["prompt"])
    input_ids_list = input_ids[0].tolist()
    logits = model.get_logits_from_input_ids(input_ids_list)
    print(f"Number of logits: {len(logits)}")
    print(logits[:10])
