from pathlib import Path
from src.parser import parse_json

print("Call Me Maybe started!")

functions = parse_json(Path("data/input/functions_definition.json"))
promts = parse_json(Path("data/input/function_calling_tests.json"))


print(f"Loaded {len(functions)} functions.")
print(f"Loaded {len(promts)} promts.")