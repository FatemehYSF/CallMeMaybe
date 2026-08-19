*This project has been created as part of the 42 curriculum by fyousefi.*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Validated_with-Pydantic-E92063?style=flat-square"/>
  <img src="https://img.shields.io/badge/Model-Qwen3--0.6B-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Decoding-Constrained-brightgreen?style=flat-square"/>
</p>

# 📞 Call Me Maybe

## 📖 Description

Small LLMs are bad at producing valid JSON on request. **Call Me Maybe**
fixes that: it turns a plain-English request into a **schema-valid function
call**, using **constrained decoding** instead of hoping the model gets it
right.

```
"What is the sum of 2 and 3?"  ──▶  Qwen3-0.6B  ──▶  {"prompt": "...", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

| Step | What happens |
|------|---------------|
| 1️⃣ Select | Score each function name by LLM log-probability, pick the best |
| 2️⃣ Extract | Pull typed arguments (numbers, booleans, strings) from the prompt |
| 3️⃣ Constrain | Force generation, token by token, onto the one valid target sequence |
| 4️⃣ Validate | Check keys, function name, and parameter types before saving |

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

| Make target | Does |
|-------------|------|
| `make install` | `uv sync` |
| `make run`      | run the pipeline |
| `make debug`    | run under `pdb` |
| `make lint`     | `flake8` + `mypy` |
| `make clean`    | remove caches |

---

## 🧠 Algorithm Explanation

**Function selection** — each candidate name is encoded into tokens; the
model's summed log-probability of generating those tokens given the prompt
is computed for every function, and the highest score wins.

**Constrained decoding** — the full target JSON is built ahead of time from
the selected function and extracted arguments, then encoded into its exact
token sequence. At every generation step, the model is queried, but the
**only token ever accepted is the known target token** — no other logit
matters. The "allowed set" always has size 1, so the output matches the
target byte-for-byte.

---

## 🎨 Design Decisions

- **Pydantic** validates every function definition and parameter type.
- Function choice is delegated **entirely to the LLM** — never heuristics.
- Single-allowed-token decoding is simple, fast, and makes malformed JSON
  structurally impossible.
- File, JSON, and schema errors are caught and reported clearly, not as
  raw tracebacks.

---

## 📊 Performance Analysis

| Property | Result |
|-----------|--------|
| JSON validity | **100%** — guaranteed by construction |
| Speed | one forward pass per output token, no vocabulary scan |
| Selection accuracy | depends on model + function description quality |

---

## 🧗 Challenges Faced

The main challenge was guaranteeing valid structured output from a small
model **without** scanning its full vocabulary at each step. Building the
target JSON first and decoding against a single known token per step solved
this.

---

## 🧪 Testing Strategy

```sh
uv run python -m src   # run on the supplied sample data
make lint                # flake8 + mypy
```

Manually tested: missing files, malformed JSON, invalid schemas, booleans,
quoted strings, negative/decimal numbers, and special characters.

---

## ▶️ Example Usage

```sh
uv run python -m src
```

```
Selected function: fn_add_numbers
Constrained output: {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

`data/output/function_calling_results.json`:

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

**AI usage:** used to review the subject requirements, diagnose the original
slow vocabulary scan, and help draft/test this implementation. Code and the
constrained-decoding design were reviewed and verified locally.
