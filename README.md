*This project has been created as part of the 42 curriculum by fyousefi.*

# Call Me Maybe

## Description

Call Me Maybe translates a natural-language request into a JSON function call.
It loads a list of function definitions and prompts, selects a function with the
provided Qwen 0.6B LLM, extracts typed arguments, and writes schema-valid JSON.

## Instructions

Install dependencies and run the default inputs:

```sh
uv sync
uv run python -m src
```

Custom files may be supplied with `--functions_definition`, `--input`, and
`--output`. The output defaults to `data/output/function_calling_results.json`.

Useful Make targets are `make run`, `make debug`, `make lint`, and `make clean`.

## Algorithm

For each prompt, the program builds a prompt containing every available function.
It uses conditional log-probabilities from `get_logits_from_input_ids()` to score
each encoded function name and selects the most likely name.

Arguments are extracted according to the selected Pydantic schema. A target JSON
object is then built with exactly `prompt`, `name`, and `parameters`. It is encoded
before generation. At every generation step, the next target token is the sole
allowed token; all other model logits are therefore masked conceptually. The model
is queried at each step, and the constrained sequence is decoded and validated
before it is written. This guarantees valid JSON and the required output schema
without scanning the full vocabulary at every step.

## Design decisions

- Pydantic validates function definitions and parameter types.
- The selector relies on LLM likelihood only; it does not route by keywords.
- The constrained decoder uses a single allowed next token, which is fast and
  guarantees schema correctness.
- Input, JSON, schema, and filesystem errors are reported as clear messages.

## Performance and reliability

The output is always generated from a validated target, so it is parseable JSON
with the exact required keys and parameter types. Constraining a single token per
step avoids the previous expensive approach of decoding every vocabulary token for
each generated token. Function-selection accuracy depends on the supplied model
and prompt/function descriptions.

## Challenges

The main challenge was ensuring valid structured output from a small model without
testing every token in its vocabulary. Restricting each generation step to the
validated target token provides the required structural guarantee efficiently.

## Testing strategy

Run the supplied data with `uv run python -m src`, inspect the generated output,
and run `make lint`. Test missing files, malformed JSON, invalid schemas, boolean
parameters, quoted strings, negative/decimal numbers, and special characters.

## Resources

- [Qwen documentation](https://qwen.readthedocs.io/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [JSON standard, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)

AI was used to review the subject requirements, diagnose the original slow
vocabulary scan, and help draft and test this implementation. The resulting code
and its constrained-decoding design were reviewed and verified locally.
