# nn

The stateless neural-network operations every webtyp model computes (matrix multiplication,
normalization, activations, softmax, rotary position embeddings), in plain Go for TinyGo/WASM.
They are used by `webtyp/encoder`, `webtyp/decoder` and the speech models, so each operation
is written once.

> **STATUS (remove this note when v0.1.0 is published):** the functions move here from
> `webtyp/transformer` (now `webtyp/encoder`) in `docs/PLAN.md`.

## Documentation

- [Agent guide](AGENTS.md): rules for anyone changing this library.
