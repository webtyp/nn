# nn
<img src="docs/img/badges.svg">

The stateless neural-network operations every webtyp model computes (matrix multiplication,
normalization, activations, softmax, rotary position embeddings), in plain Go for TinyGo/WASM.
They are used by `webtyp/encoder`, `webtyp/decoder` and the speech models, so each operation
is written once.

| I want... | Use... |
|---|---|
| Matrix multiplication (transposed B) | `nn.MatmulT` |
| Layer normalization | `nn.LayerNorm` |
| RMS normalization | `nn.RMSNorm` |
| Softmax | `nn.Softmax` |
| GELU activation | `nn.GELU` |
| SiLU activation | `nn.SiLU` |
| Element-wise addition | `nn.Add` |
| Rotary position embedding | `nn.RoPE` |

## Documentation

- [SIMD](docs/SIMD.md): what SIMD is, the measured 3.3× speed-up and the three conditions it needs, and how the framework ships a SIMD and a non-SIMD worker.
- [Agent guide](AGENTS.md): rules for anyone changing this library.
