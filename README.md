*This project has been created as part of the 42 curriculum by fyousefi.*

# Call Me Maybe

Turns a natural-language prompt into a valid JSON function call using
**Qwen3-0.6B** and **constrained decoding**.

```
"What is the sum of 2 and 3?"  ->  {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

## Description

Small LLMs often produce invalid JSON: missing brackets, numbers in quotes,
extra text. This project lets the model make every decision (which function,
which values) but only allows it to emit tokens that keep the output valid
for the target schema.

| File | Role |
|------|------|
| `data/input/functions_definition.json` | Available functions: name, description, typed parameters |
| `data/input/function_calling_tests.json` | Prompts to process |
| `data/output/function_calling_results.json` | One `{prompt, name, parameters}` object per prompt |

## Instructions

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

| Command | Action |
|---------|--------|
| `make install` | Install dependencies (`uv sync`) |
| `make run` | Run with the default input/output paths |
| `make debug` | Run under `pdb` |
| `make lint` | `flake8` + strict `mypy` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |

Custom paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

## Algorithm

Each prompt goes through four stages:

```
select function -> build argument prompt -> constrained decoding -> validate
```

### 1. Function selection (`function_selector.py`)

The selection prompt lists every function with its description and
parameters. Instead of free generation, each candidate name is scored by
its sequence log-probability:

```
score(name) = sum_i log P(token_i | prompt, token_<i)
```

The highest score wins. Only existing names are scored, so the model cannot
invent a function.

### 2. Argument prompt (`prompt_builder.py`)

A second prompt describes only the selected function and its typed
parameters. Three few-shot examples on unrelated functions teach the
word-to-literal mapping (e.g. "seven" -> `7.0`, "hashes" -> `#`) without
leaking test answers. An empty `<think></think>` block disables Qwen3's
reasoning preamble.

### 3. Constrained decoding (`constrained_decoder.py`)

The JSON skeleton (keys, quotes, braces, commas) is emitted by the code. The
model generates only the values. At each step the logits over the ~151k-token
vocabulary are masked so only type-valid tokens remain, then the best one is
taken (greedy).

| Type | Allowed tokens | Termination |
|------|----------------|-------------|
| `string` | Tokens that keep a valid JSON string, including escapes (`\\d`) | A token whose first unescaped `"` closes the string |
| `number` | Digits, `-`, `.` forming a valid numeric prefix | `,` or `}` scores higher than any continuation |
| `integer` | Digits and `-` | Same as `number` |
| `boolean` | `true` / `false` | Higher log-probability wins |

The vocabulary is decoded once at startup and grouped by category
(plain text, escapes, quote-containing, digits), so each mask is a NumPy
index lookup rather than a scan over all tokens.

### 4. Validation (`validator.py`)

Checks that keys are exactly `prompt`, `name`, `parameters`, that the name
matches the selected function, that parameter names match the definition and
that every value has the declared type (`bool` is rejected for numbers).
A failing prompt is reported and skipped; other results are still saved.

## Modules

| Module | Responsibility |
|--------|----------------|
| `__main__.py` | CLI, loads the model once, main loop, writes output |
| `parser.py` | Reads input JSON and builds function models |
| `models.py` | Pydantic `Parameter` / `FunctionDefinition` |
| `vocabulary.py` | Loads the tokenizer vocabulary, token -> id map |
| `prompt_builder.py` | Selection and argument prompts |
| `function_selector.py` | Log-probability scoring of function names |
| `constrained_decoder.py` | Token masking and value generation |
| `validator.py` | Schema check of the final call |
| `llm_sdk/` *(provided)* | `encode`, `decode`, `get_logits_from_input_ids` |

## Design Decisions

- **Model decides, code constrains:** no regex extraction from the prompt, so
  new functions and phrasings work without code changes.
- **Forced skeleton:** keys and punctuation never cost a model decision and
  can never be malformed.
- **Pydantic input models:** malformed definition files fail early with a
  clear error.
- **Per-prompt error handling:** one failure does not discard the batch.

## Performance

On the 11 sample prompts (Apple GPU, ~40–60 s total):

| Metric | Result |
|--------|--------|
| JSON validity | 100% (guaranteed by masking) |
| Function selection | 11 / 11 |
| Exact argument values | 10 / 11 |

Each generated token costs one full forward pass, since the SDK has no
KV cache.

**Known limitation:** for "Replace all vowels ... with asterisks" the model
writes `"asterisk"` instead of `"*"` (`*` is its second choice). This is
intentionally not hardcoded; a larger model or multi-candidate sampling
would fix it.

## Challenges

| Problem | Solution |
|---------|----------|
| Masking 151k tokens per step was slow | Pre-group the vocabulary once, mask with NumPy |
| Merged tokens such as `",` close a string mid-token | Treat any token whose first quote closes the string as a terminator |
| Regex arguments need backslashes | Allow JSON escape sequences inside strings |
| Similar functions score closely | Include parameters in the selection prompt to widen the margin |

## Testing

`make run` on the sample data, `make lint` for style and types. Edge cases
checked manually: missing files, malformed JSON, invalid schemas, booleans,
negative and decimal numbers, special characters, and paraphrased prompts to
measure selection margins.

Sample output of `make run`:

```
fn_add_numbers -> -6.25e-05
fn_greet -> -14.09
...
Selected function: fn_add_numbers
Constrained output: {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

## Resources

- [Qwen documentation](https://qwen.readthedocs.io/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [RFC 8259 — JSON](https://www.rfc-editor.org/rfc/rfc8259)

**AI usage:** AI was used to review the subject requirements, find the
bottleneck in the original vocabulary scan, replace regex argument extraction
with masked constrained decoding, and help draft this README. All code and the
decoding design were reviewed and verified locally.
