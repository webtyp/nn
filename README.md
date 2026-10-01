# nn
<img src="docs/img/badges.svg">

The stateless neural-network operations every webtyp model computes (matrix multiplication,
normalization, activations, softmax, rotary position embeddings), in plain Go for TinyGo/WASM.
They are used by `webtyp/encoder`, `webtyp/decoder` and the speech models, so each operation
is written once.

| I want... | Use... |
|---|---|
| Matrix multiplication (transposed B) | `nn.MatmulT` |
| Matrix × vector with int8 weights in blocks of 32 (GGUF Q8_0) | `nn.MatVecInt8Block32` |
| The same for many inputs at once (reading a prompt) | `nn.MatmulInt8Block32` |
| Matrix × vector with int8 weights AND int8 input (vectorizes in WebAssembly, 4.3× with SIMD) | `nn.QuantizeBlocks32` + `nn.MatVecQ8Block32` |
| Layer normalization | `nn.LayerNorm` |
| RMS normalization | `nn.RMSNorm` |
| Softmax | `nn.Softmax` |
| GELU activation | `nn.GELU` |
| SiLU activation | `nn.SiLU` |
| Element-wise addition | `nn.Add` |
| Rotary position embedding | `nn.RoPE` |

## Documentation

- [SIMD](docs/SIMD.md): what SIMD is, the measured 3.3× speed-up and the three conditions it needs, and how the framework ships a SIMD and a non-SIMD worker.
- [Performance audit](docs/PERFORMANCE.md): where the time goes, comparison with llama.cpp and browser runtimes, the way out
- [Agent guide](AGENTS.md): rules for anyone changing this library.
