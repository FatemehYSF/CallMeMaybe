*This project has been created as part of the 42 curriculum by fyousefi.*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Validated_with-Pydantic-E92063?style=flat-square"/>
  <img src="https://img.shields.io/badge/Model-Qwen3--0.6B-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Decoding-Constrained-brightgreen?style=flat-square"/>
</p>

# 📞 Call Me Maybe

## 📖 Description

Small language models are bad at producing valid JSON on request. **Call Me
Maybe** fixes that: it turns a plain-English request into a **schema-valid
function call**, using **constrained decoding** instead of hoping the model
gets it right.

```
"What is the sum of 2 and 3?"
        │
        ▼
┌───────────────────┐        {
│   Qwen3-0.6B (LLM) │  ───▶    "prompt": "What is the sum of 2 and 3?",
└───────────────────┘          "name": "fn_add_numbers",
                                "parameters": {"a": 2.0, "b": 3.0}
                              }
```

| Step | What happens |
|------|---------------|
| 1️⃣ Select | Score every function name by LLM log-probability, pick the best |
| 2️⃣ Extract | Pull typed arguments (numbers, booleans, strings) from the prompt |
| 3️⃣ Constrain | Force generation, token by token, onto the one valid target sequence |
| 4️⃣ Validate | Check keys, function name, and parameter types before saving |

Output is **always** parseable JSON with the exact required keys — schema
violations are structurally impossible.

---

## ⚙️ Instructions

```sh
uv sync
uv run python -m src
```

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

By default, inputs are read from `data/input/` and results are written to
`data/output/function_calling_results.json`.

| Make target | Does |
|-------------|------|
| `make install` | `uv sync` |
| `make run`      | run the pipeline |
| `make debug`    | run under `pdb` |
| `make lint`     | `flake8` + `mypy` |
| `make clean`    | remove caches |

---

## 🧠 Algorithm Explanation

**Function selection** — every candidate function name is encoded into
tokens; the model's summed log-probability of generating those exact tokens
(given the prompt) is computed for each. The highest-scoring name wins.
No keyword matching, purely LLM likelihood.

**Constrained decoding** — the full target JSON object
(`prompt` / `name` / `parameters`) is built ahead of time from the selection
and extraction steps, then encoded into its exact token sequence. At every
generation step, the model is queried, but the **only token ever accepted is
the known target token** — every other logit is irrelevant. This is the
strictest form of constrained decoding: the "allowed set" always has size 1,
so the decoded output is guaranteed to match the target byte-for-byte.

```
model.get_logits_from_input_ids(context)  →  [target_token]  →  append  →  repeat
```

---

## 🎨 Design Decisions

- **Pydantic** validates every function definition and parameter type.
- Function choice is delegated **entirely to the LLM** — never heuristics.
- Single-token-allowed decoding is simple, fast, and removes any possibility
  of malformed JSON reaching the output file.
- All file, JSON, and schema errors are caught and reported with a clear
  message instead of a raw traceback.

---

## 📊 Performance Analysis

| Property | Result |
|-----------|--------|
| JSON validity | **100%** — guaranteed by construction |
| Speed | one forward pass per output token, no vocabulary scan |
| Selection accuracy | depends on model + function description quality |

Restricting each step to a single allowed token replaced an earlier,
much slower approach that re-scored the entire vocabulary at every step.

---

## 🧗 Challenges Faced

The core challenge was **guaranteeing valid structured output from a small
model without scanning its full vocabulary** at each generation step.
Building the target JSON first and decoding against a single known token per
step solved this: the model can no longer "wander off" schema.

---

## 🧪 Testing Strategy

```sh
uv run python -m src   # run on the supplied sample data
make lint                # flake8 + mypy
```

Manually exercised: missing files, malformed JSON, invalid schemas, boolean
parameters, quoted strings, negative/decimal numbers, and special characters.

---

## ▶️ Example Usage

```sh
uv run python -m src
```

```
Call Me Maybe started!
Selected function: fn_add_numbers

user: What is the sum of 2 and 3?
Selected function: fn_add_numbers
Constrained output: {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
Arguments: [2.0, 3.0]
```

Result written to `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

---

## 📚 Resources

- [Qwen documentation](https://qwen.readthedocs.io/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [JSON standard — RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)

**AI usage:** AI was used to review the subject requirements, diagnose the
original slow vocabulary scan, and help draft and test this implementation.
The resulting code and its constrained-decoding design were reviewed and
verified locally.
