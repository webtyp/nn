---
PLAN: "feat: nn — neural-network operations moved out of transformer (MatmulT, norms, activations, Softmax, RoPE, Add)"
TAG: v0.1.0
EXECUTOR: jules
REVIEWER: none
STATUS: review
SESSION: 13142100057473317889
PR: https://github.com/webtyp/nn/pull/1
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
>
> Part of
> [`AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md).
> Independent. `webtyp/encoder` (the rename of `webtyp/transformer`) and `webtyp/decoder`
> wait for this tag.

# Plan — `webtyp.com/nn`: the operations every model in webtyp computes

## 0. Context

A neural network in webtyp is a sequence of the same few operations: multiply by a weight
matrix, normalize, apply an activation, softmax, rotate positions (RoPE). Today those
functions live in `kernels.go` of **`webtyp/transformer`**. That repository is being renamed
to `webtyp/encoder`, because it only computes the encoder of an embedding model.

Three more models are coming, and all three need the same functions:
- the LLM decoder (`webtyp/decoder`, for Qwen3.5),
- speech-to-text,
- text-to-speech.

If each copied `kernels.go`, the browser binary would carry the same code several times, and
a fix would have to land in four places. So the functions move to their own repository, and
every model imports them.

This plan **copies** the functions and their tests into this repository, unchanged except for
the package name. It does **not** touch `webtyp/encoder`. Deleting the old copy there is the
next plan (encoder), which waits for this tag.

The source files are attached below (§ Source), because the executor only has this repository.

## Development rules (inline)

- **Primary runtime: browser, TinyGo/WASM.** Every file compiles under `GOOS=js GOARCH=wasm`
  and TinyGo.
- **Plain Go, one implementation.** No SIMD, no assembly, no build tags. The same code runs
  on every target.
- **Never import:** `fmt`, `errors`, `strings`, `strconv` (use `webtyp.com/fmt`), `context`,
  `encoding/json`, `sort`, `map[K]V`, `os`, `log`. **`math` is allowed**, as it already is in
  the source.
- Dependencies: `webtyp.com/fmt` and `webtyp.com/vector` (for `vector.Dot`, the inner loop of
  `MatmulT`), the same two the source uses.
- Behaviour must not change: the encoder was verified to cosine ≥ 0.999 against the real
  model with these exact functions.
- Tests: `testing` + `math` only. Do **not** run `gopush`/`codejob`.

## Design gate (api-design — five answers)

1. **Prior art.** **PyTorch `torch.nn.functional`** (`F.layer_norm`, `F.silu`,
   `F.softmax`), **JAX `jax.nn`** (`jax.nn.gelu`, `jax.nn.softmax`) and **TensorFlow
   `tf.nn`** (`tf.nn.silu`, `tf.nn.softmax`) all group the stateless operations of a network
   in a package called `nn`, separate from any model. **ggml** (llama.cpp) keeps them in one
   library used by every architecture. We follow the same split. The difference is that ours
   work on plain `[]float32` with explicit shapes and no tensor type, because TinyGo binaries
   pay for every abstraction.
2. **Novice-name test.** `nn.RMSNorm(dst, src, gamma, dim, eps)` and `nn.MatmulT(dst, a, bT, m, k, n)`
   read as what they compute. "nn" is the universal abbreviation of "neural network" in every
   ML framework. The function names do not change, so `encoder` only changes its import.
3. **Complexity ledger.**
   ```
   Concepts the developer must learn   +0 / −0   (the same 8 functions, new import path)
   Files they must touch to do X       +0 / −0
   Lines at the call site              +0 / −0   (transformer.X → nn.X)
   Ways to do the same thing           +1 / −1   (+1 here until the encoder plan deletes kernels.go)
   ```
4. **Where it belongs.** They are stateless math on slices, shared by every model. They are
   not part of an encoder, a decoder or a speech model, so they get their own repository.
   `GatedFFN` and `Activation` stay in `encoder`. They pack the gate and up projections in one
   matrix, which is ModernBERT's layout and not a shared operation.
5. **What it deletes.** Nothing in this repository. The encoder plan deletes `kernels.go` and
   the kernel tests from `webtyp/encoder`.

## Stage 1 — the functions

Split the source (§ Source, `kernels.go`) into these files, **package `nn`**, bodies
unchanged:

| File | Functions |
|---|---|
| `matmul.go` | `MatmulT` |
| `norm.go` | `LayerNorm`, `RMSNorm` |
| `activation.go` | `GELU`, `SiLU` |
| `softmax.go` | `Softmax` |
| `rope.go` | `RoPE` |
| `add.go` | `Add` |

Error messages change their prefix from `transformer:` to `nn:` (for example
`fmt.Err("nn: invalid matmul dimensions")`). Otherwise the text stays the same. Add a package
doc comment in `matmul.go`:

```go
// Package nn holds the stateless operations every webtyp model computes: matrix
// multiplication, normalization, activations, softmax and rotary position embeddings. They work
// on plain []float32 slices with explicit shapes, in plain Go, for TinyGo/WASM.
package nn
```

`go.mod`: `go get webtyp.com/fmt@v1.0.0 webtyp.com/vector@v0.1.1`.

## Stage 2 — tests

Copy the source tests (§ Source, `kernels_test.go`) into files named after Stage 1
(`matmul_test.go`, `norm_test.go`, `activation_test.go`, `softmax_test.go`, `rope_test.go`,
`add_test.go`), `package nn`. Keep all 12 tests with their exact names and tolerances. Change
only the package clause and any error-prefix assertion.

## Stage 3 — docs

`README.md`: remove the `STATUS` note; keep one paragraph (what `nn` is), add an "I want X → use Y" table with the 8 functions,
and a link to `AGENTS.md`. Remove nothing else.

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | the 6 files above, `go.mod` | `grep -rn '"fmt"\|"strings"\|"errors"\|map\[' --include=*.go .` → empty; `grep -rn "transformer" --include=*.go .` → empty |
| 2 | 6 test files | the 12 tests pass under `gotest` and `gotest -tinygo` |
| 3 | `README.md` | lists the 8 functions |

## Source

Copy from `webtyp/encoder` (formerly `webtyp/transformer`), `main` branch:
`https://github.com/webtyp/encoder/blob/main/kernels.go` and
`https://github.com/webtyp/encoder/blob/main/kernels_test.go`. Both files are public. If you
cannot fetch them, stop and report it. Do not re-write them from memory, because the numeric
behaviour (float64 accumulation, NeoX-style RoPE) is what was verified.
