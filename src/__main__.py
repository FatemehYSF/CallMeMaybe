import argparse
import json
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from src.argument_exctractor import extract_arguments
from src.constrained_decoder import generate_constrained_output
from src.function_selector import select_function
from src.parser import build_function_models, parse_json
from src.prompt_builder import build_prompt
from src.validator import validate_function_call


def parse_arguments() -> argparse.Namespace:
    """Get file paths from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default=(
            "data/input/functions_definition.json"
        ),
    )

    parser.add_argument(
        "--input",
        default=(
            "data/input/function_calling_tests.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/output/"
            "function_calling_results.json"
        ),
    )

    return parser.parse_args()


def load_input_file(
    path: Path,
) -> list[dict[str, Any]]:
    """Open and read a JSON input file."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    try:
        return parse_json(path)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in file: {path}"
        ) from error


def main() -> None:
    """Read prompts and create function calls."""
    args = parse_arguments()

    print("Call Me Maybe started!")

    model = Small_LLM_Model()

    functions = load_input_file(
        Path(args.functions_definition)
    )

    prompts = load_input_file(
        Path(args.input)
    )

    function_models = build_function_models(
        functions
    )

    results: list[dict[str, Any]] = []

    for prompt_data in prompts:
        user_prompt = prompt_data["prompt"]

        full_prompt = build_prompt(
            function_models,
            user_prompt,
        )

        input_ids = model.encode(
            full_prompt
        )[0].tolist()

        selected_function = select_function(
            model,
            input_ids,
            function_models,
        )

        function = next(
            function
            for function in function_models
            if function.name == selected_function
        )

        print(
            f"Selected function: {selected_function}"
        )

        arguments = extract_arguments(
            user_prompt,
            function,
        )

        generated_tokens = generate_constrained_output(
            model,
            input_ids,
            user_prompt,
            function,
            arguments,
        )

        generated_text = model.decode(
            generated_tokens
        )
        result = json.loads(generated_text)

        validate_function_call(
            result,
            function,
        )

        results.append(result)

        print()
        print(f"user: {user_prompt}")
        print(f"Selected function: {selected_function}")
        print(f"Constrained output: {generated_text}")
        print(f"Arguments: {arguments}")
        print()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        ValidationError,
    ) as error:
        print(f"Error: {error}")
