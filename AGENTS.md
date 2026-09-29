# Agent Guide — `webtyp/nn`

Constraints for agents working on this library. **Read this before any change.**
The current work order, when one exists, is [docs/PLAN.md](docs/PLAN.md). The ecosystem plan
is [`agent/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md).

## What this library is

The stateless operations every webtyp model computes (matmul, norms, activations, softmax,
RoPE), on plain `[]float32` slices with explicit shapes. It is used by `encoder`, `decoder` and
the speech models. It never contains a model, a layer with weights, or a graph. Those belong to
the repository of the model that uses them.

**Its primary runtime is a browser tab compiled with TinyGo.** Speed here is the speed of every
model in the ecosystem, and correctness here is verified numerically against real models, so
never change a function's arithmetic without re-running the model-level reference tests.

## The builds that define "done"

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
```

## Never import these

| Never | Use instead | Why |
|---|---|---|
| `fmt`, `errors`, `strings`, `strconv` | `webtyp.com/fmt` | isomorphic, small under TinyGo |
| `sort`, `encoding/json` | nothing | reflection is a size tax under TinyGo |
| `map[K]V` | a slice | TinyGo's map runtime is a size tax |
| `os`, `log`, `net/http` | nothing | pure math never touches the environment |

`math` is allowed.

## Common mistakes to avoid

- Adding a second implementation of a function (SIMD, assembly, build tags). For now there is
  one plain-Go implementation for every target. Changing that is an open decision of the
  ecosystem master plan.
- Allocating inside a function. Callers pass `dst`, which keeps the garbage collector quiet
  during inference.
- Accumulating sums in float32. Norms accumulate in float64 on purpose.
