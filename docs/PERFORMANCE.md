# Performance audit — why the in-browser model is slow, and the way out

This page explains how fast webtyp's own model runtime is, where its time goes, and how it
compares with other implementations that run models in a browser. Read it before optimizing
`nn`, `decoder` or `qwen`: every number here was measured on 2026-09-30 on the developer laptop
(Intel i7-11800H, 8 cores / 16 threads, RTX 3060 Laptop GPU), with the real Qwen3.5-0.8B int8.

## Where we are

| Runtime | Where | Decode speed |
|---|---|---|
| **webtyp** (`qwen` + `decoder` + `nn`), native Go, 1 thread, no SIMD | CPU | **~1.9 tok/s** (0.53 s/token) |
| llama.cpp, Q8_0, 1 thread (AVX2) | CPU | 6.7 tok/s |
| llama.cpp, Q8_0, 8 threads | CPU | 23.8 tok/s |
| WebLLM (WebGPU), Qwen2.5-0.5B q4f16, Chrome | GPU | 46–51 tok/s ([arXiv 2604.02344](https://arxiv.org/html/2604.02344)) |
| WebLLM, Qwen2.5-1.5B | GPU | 30–46 tok/s (same source) |

Other browser runtimes: [wllama](https://github.com/ngxson/wllama) is llama.cpp compiled to
WebAssembly, with SIMD, multithreading (it needs the `Cross-Origin-Opener-Policy` and
`Cross-Origin-Embedder-Policy` headers) and WebGPU. transformers.js runs ONNX on WASM or WebGPU.
We found no published per-token numbers for them on these models.

So we are **3.5× slower than llama.cpp on one thread**, 12× slower than llama.cpp on eight, and
about 25× slower than the best browser runtime on a GPU.

## Where the time goes

A CPU profile of 20 decoder steps (`go tool pprof`):

| Function | Share |
|---|---|
| `nn.MatVecInt8Block32` (matrix × vector, int8 weights) | **96 %** |
| `decoder.deltaRuleStep` (DeltaNet recurrence) | 3 % |
| everything else | 1 % |

The model is a chain of matrix-vector products, about 500 M multiply-adds per token. Nothing else
matters until that kernel is fast.

## What we tried

| Change | Result |
|---|---|
| Eight independent accumulators, re-sliced fixed-length views (no bounds checks) | **1.4×**, shipped in `nn` v0.3.1 |
| Reading the prompt as a batch (`MatmulInt8Block32`, each weight row read once for 32 inputs) | **no gain** (78 ms vs 78 ms): the cost is arithmetic, not memory |
| int8 × int8 with integer accumulation (llama.cpp's Q8_0 · Q8_0) | slower in **native** Go, but 2.1–4.3× faster in WebAssembly, the real target (next section) |
| Not reading the same prefix twice (`qwen` v0.2.0 prefix cache) | second turn 21.8 s instead of 71.5 s |

Those measurements were native Go. The browser runs TinyGo's WebAssembly, whose compiler (LLVM)
behaves differently, and the next section is what it showed.

## What we were doing wrong (found 2026-10-01)

Two defects, both measured.

**1. The kernel's form prevented SIMD.** `MatVecInt8Block32` converts each int8 weight to float and
adds the products into a float sum. A compiler may not reorder float additions (the result would
change), so LLVM cannot vectorize that loop: with `simd128` enabled it ran exactly as fast as
without. The fix is the form llama.cpp uses. Quantize the input vector **once per token** to int8
in blocks of 32 (`nn.QuantizeBlocks32`), then make each block an **integer** dot product scaled by
the two block scales (`nn.MatVecQ8Block32`). Integer additions can be reordered, so LLVM
vectorizes them. The error is 0.3 % relative RMS, from quantizing the input.

Measured under TinyGo 0.41 (`-opt=2`, Node 22), one 3584×1024 int8 matrix:

| Kernel | WebAssembly, no SIMD | WebAssembly + SIMD128 | native Go |
|---|---|---|---|
| float accumulation (`MatVecInt8Block32`, today) | 2.6–3.0 ms | 2.6 ms (SIMD unused) | 2.0 ms |
| int8 × int8, one row at a time | 1.5 ms | 0.63 ms | 3.3 ms |
| **int8 × int8, two rows at a time (`MatVecQ8Block32`)** | **1.30 ms (2.1×)** | **0.61 ms (4.3×)** | — |
| int8 lookup table instead of conversion | 2.4 ms | 2.4 ms | 2.0 ms |
| int16 pair products | 1.6 ms | 1.6 ms (breaks vectorization) | — |

It wins even without SIMD, on any browser. Native Go is slower with it (the Go compiler does not
vectorize), and native Go is only used for tests on the developer machine.

**2. The output projection was computed for every prompt token.** `decoder.Step` always
multiplies by the full vocabulary (the tied embedding: 248 320 × 1024 for Qwen3.5) and fills the
logits, even for prompt tokens whose logits are thrown away. That is **34 % of every step** for
Qwen3.5-0.8B (254 M of ~750 M multiply-adds) and 19 % for LFM2.5-350M. A decision needs only the
2–10 option letters at the last position, not 248 320 logits. The fix belongs to `decoder`: read a
token without computing logits, and compute only the rows asked for.

## Tiers: the agent must work on every device

The runtime is built so that every device runs the agent, and better hardware only makes it
faster:

| Tier | Needs | Kernel | Expected for one decision (~50 new tokens, decider-0.8b) |
|---|---|---|---|
| **1. baseline** | any browser with WebAssembly | `MatVecQ8Block32` in a plain `-opt=2` build | ≈ 5–9 s |
| **2. SIMD** | SIMD128 (Chrome 91+, Firefox 89+, Safari 16.4+) | the same code, `+simd128` build | ≈ 2–4 s |
| **3. WebGPU** | WebGPU and a usable GPU | compute shaders | under 1 s (WebLLM-class) |

Estimates from the kernel numbers above with defect 2 fixed (prompt tokens without the output
projection). They are to be confirmed end to end. The Worker picks the highest tier the browser
supports (a feature test, as decision D16 already does for SIMD), and every tier runs the same
model and gives the same decisions within quantization noise.

## The way out, in order of cost

1. **SIMD128 in WebAssembly**, with the integer kernel above: **4.3×** on int8 matrices. It needs
   TinyGo `-opt=2` and a target with `+simd128` (`testdata/wasm-simd.json`); the integer kernel
   needs no special layout. Browsers from 2021 on have SIMD128. The Worker build ships both a SIMD
   and a plain binary (decision D16), and the plain one is tier 1.
2. **Several cores.** llama.cpp gains 3.5× from 1 to 8 threads. In a browser, that means Web
   Workers sharing the weights through a `SharedArrayBuffer`, which requires the COOP/COEP headers.
   TinyGo's WASM output has no threads, so the work must be split by rows across Workers that read
   one shared buffer. Copying the weights into each Worker is not possible on 4 GB machines.
3. **WebGPU.** This is the path the fastest browser runtimes use (40–50 tok/s). The kernels become
   compute shaders (WGSL) dispatched through the browser's WebGPU API from Go (`syscall/js`, via
   `webtyp/js`). It is the largest change and the largest gain, and it depends on the clinic PCs'
   integrated GPUs supporting WebGPU.

The hybrid agent needs one forward pass per decision and no generation for most answers
(`agent/docs/HYBRID_DESIGN.md`), so with SIMD a decision over ~50 new tokens would take a few
seconds instead of ~25 s.
