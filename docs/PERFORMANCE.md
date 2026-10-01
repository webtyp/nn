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
| int8 × int8 with integer accumulation (llama.cpp's Q8_0 · Q8_0) | slower in scalar Go, and less exact |
| Not reading the same prefix twice (`qwen` v0.2.0 prefix cache) | second turn 21.8 s instead of 71.5 s |

Scalar Go has reached its limit. The rest must come from doing several multiply-adds per
instruction, or on several cores, or on the GPU.

## The way out, in order of cost

1. **SIMD128 in WebAssembly.** It has been measured at **3.3×** (`docs/SIMD.md`), and it needs
   TinyGo `-opt=2`, the `simd128` target and loops in axpy form. That would put us around llama.cpp
   on one thread (~6 tok/s). Every browser has SIMD128 today. The Worker build ships both a SIMD
   and a plain binary (decision D16).
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
