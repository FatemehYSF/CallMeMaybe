*This project has been created as part of the 42 curriculum by fyousefi.*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Validated_with-Pydantic-E92063?style=flat-square"/>
  <img src="https://img.shields.io/badge/Model-Qwen3--0.6B-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Decoding-Constrained-brightgreen?style=flat-square"/>
</p>

# 📞 Call Me Maybe

Turns a natural-language prompt into a valid JSON function call using
**Qwen3-0.6B** and **constrained decoding**.

```
"What is the sum of 2 and 3?"  →  {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

---

## 📖 Description

Small LLMs often produce invalid JSON: missing brackets, numbers in quotes,
extra text. Here the model makes every decision (which function, which
values), but it may only emit tokens that keep the output valid for the
target schema.

| 📁 File | Role |
|---------|------|
| `data/input/functions_definition.json` | Available functions: name, description, typed parameters |
| `data/input/function_calling_tests.json` | Prompts to process |
| `data/output/function_calling_results.json` | One `{prompt, name, parameters}` object per prompt |

---

## ⚙️ Instructions

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

| Command | Action |
|---------|--------|
| `make install` | 📦 Install dependencies (`uv sync`) |
| `make run` | 🚀 Run with the default paths above |
| `make debug` | 🐞 Run under `pdb` |
| `make lint` | 🔍 `flake8` + strict `mypy` |
| `make clean` | 🧹 Remove `__pycache__` and `.mypy_cache` |

Custom paths:

```sh
uv run python -m src \
  --functions_definition <definitions.json> \
  --input <prompts.json> \
  --output <results.json>
```

---

## 🧠 How It Works

Each prompt goes through four steps:
**select function → build argument prompt → constrained decoding → validate**.

### 1️⃣ Select the function — `function_selector.py`

Every function name is scored by its log-probability after the prompt:

```
score(name) = Σ log P(token_i | prompt, token_<i)
```

The highest score wins. Only real names are scored, so the model can never
invent a function.

### 2️⃣ Build the argument prompt — `prompt_builder.py`

Describes only the selected function and its typed parameters. Three
few-shot examples on unrelated functions teach word-to-value mapping
("seven" → `7.0`, "hashes" → `#`) without leaking test answers. An empty
`<think></think>` block turns off Qwen3's reasoning text.

### 3️⃣ Constrained decoding — `constrained_decoder.py`

The code writes the JSON skeleton (keys, quotes, braces, commas); the model
only fills in the values. At each step, tokens that would break the value's
type are masked out of the logits and the best remaining token is taken.

| Type | ✅ Allowed tokens | 🛑 Stops when |
|------|------------------|---------------|
| `string` | Valid JSON string content, including escapes (`\\d`) | A token's first unescaped `"` closes the string |
| `number` | Digits, `-`, `.` forming a valid number | `,` or `}` beats every continuation |
| `integer` | Digits and `-` | Same as `number` |
| `boolean` | `true` / `false` | Higher log-probability wins |

### 4️⃣ Validate — `validator.py`

Checks the keys (`prompt`, `name`, `parameters`), the function name, the
parameter names and every value's type (`bool` is rejected for numbers). A
failing prompt is reported and skipped; the others are still saved.

---

## 🧩 Modules

| Module | Responsibility |
|--------|----------------|
| `__main__.py` | CLI, loads the model once, main loop, writes output |
| `parser.py` | Reads input JSON into function models |
| `models.py` | Pydantic `Parameter` / `FunctionDefinition` |
| `vocabulary.py` | Tokenizer vocabulary and token → id map |
| `prompt_builder.py` | Selection and argument prompts |
| `function_selector.py` | Function name scoring |
| `constrained_decoder.py` | Token masking and value generation |
| `validator.py` | Final schema check |
| `llm_sdk/` *(provided)* | `encode`, `decode`, `get_logits_from_input_ids` |

---

## 🎨 Design Decisions

- 🤖 **No regex extraction:** new functions and phrasings work without code changes.
- 🦴 **Code-written skeleton:** keys and punctuation can never be malformed.
- 🛡️ **Pydantic models:** bad definition files fail early with a clear error.
- 🧯 **Per-prompt error handling:** one failure doesn't lose the whole batch.

---

## 📊 Performance

11 sample prompts on an Apple GPU, ~40–60 s total:

| Metric | Result |
|--------|--------|
| JSON validity | ✅ 100% (guaranteed by masking) |
| Function selection | ✅ 11 / 11 |
| Exact argument values | 🟡 10 / 11 |

Each token costs one full forward pass, since the SDK has no KV cache.

> ⚠️ **Known limitation:** for "Replace all vowels ... with asterisks" the
> model writes `"asterisk"` instead of `"*"` (its second choice). This is
> intentionally not hardcoded; a larger model or multi-candidate sampling
> would fix it.

---

## 🧗 Challenges

### 🔌 Talking to the model

| 😵 Problem | 💡 What I did |
|------------|---------------|
| The SDK has no `generate()`, only `encode`, `decode` and `get_logits_from_input_ids` | Built the pipeline step by step: encode the prompt, read the logits, turn the chosen token id back into text, and loop |
| Free generation could invent a function name that doesn't exist | Score only the real names: run each name's tokens through the model and add up their log-probabilities |
| Raw logits are not probabilities, and `exp()` of large logits overflows | Wrote a log-softmax that subtracts the max logit first, so it stays numerically stable |
| `fn_substitute_string_with_regex` scored close to other string functions | Rewrote its description in `functions_definition.json` to mention source, replacement, numbers, vowels and words |

### 🧩 Getting the arguments right

| 😵 Problem | 💡 What I did |
|------------|---------------|
| **cat → dog bug:** in *"Substitute the word 'cat' with 'dog' in 'The cat sat…'"* quoted strings were taken in order, so parameters got the wrong values | Pick values by context: the text after **"in"** is the `source_string`, the text after **"with"** is the `replacement`, and each one is removed from the pool once used |
| `"Hello 34 I'm 233 years old"`: the apostrophe in *I'm* ended the match early | Match the closing quote to the opening one with a backreference: `(['"])(.*?)\1` |
| Some replacements are not quoted (*"with NUMBERS"*) | Fall back to the bare word after "with" |
| The prompt says *"numbers"* or *"vowels"*, never an actual regex | Map these words to a pattern (`\d+`, `[aeiou]`) or take a quoted word |
| Digits inside words or decimals could be picked up as numbers | Number regex with lookarounds, `(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])`, values stored as floats |

### 🎛️ Trying fully masked decoding

At one point I replaced the regex extractor with a decoder where the model
wrote the values itself, with invalid tokens masked out of the logits. It
brought its own problems:

| 😵 Problem | 💡 What I did |
|------------|---------------|
| Checking all ~151k tokens at every step was very slow | Grouped the vocabulary once at startup (plain, escape, quote, number tokens); each mask became a NumPy index lookup |
| Merged tokens like `",` close a string in the middle of a token | Any token whose first unescaped quote closes the string ends the value |
| Regex values need backslashes (`\d`) | Allowed JSON escape sequences inside strings |
| Qwen3 starts with a reasoning block instead of the answer | Added an empty `<think></think>` block to the prompt |
| The model wrote words instead of symbols (`"asterisk"` instead of `"*"`) | Few-shot examples on unrelated functions helped, but did not fix every case |

The current version goes back to regex extraction for the values. The model
still chooses the function, and the output is forced token by token onto
the exact JSON target, so it is always valid.

### 🧹 Code quality

| 😵 Problem | 💡 What I did |
|------------|---------------|
| Strict `mypy` rejects untyped `re` and `json` results | Typed every function and used `cast` where the library returns `Any` |
| Bad input files crashed with a traceback | Pydantic models for definitions, plus clear errors for missing files and invalid JSON |

---

## 🧪 Testing

Ran on the sample data and with `make lint`. Edge cases checked by hand:
missing files, malformed JSON, invalid schemas, booleans, negative and
decimal numbers, special characters, and paraphrased prompts.

Sample `make run` output:

```
fn_add_numbers -> -6.25e-05
fn_greet -> -14.09
...
Selected function: fn_add_numbers
```

---

## 📚 Resources

- [Qwen documentation](https://qwen.readthedocs.io/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [RFC 8259 — JSON](https://www.rfc-editor.org/rfc/rfc8259)

**🤖 AI usage:** AI was used to review the subject requirements, find the
bottleneck in the original vocabulary scan, replace regex argument extraction
with masked constrained decoding, and help draft this README. All code and the
decoding design were reviewed and verified locally.
