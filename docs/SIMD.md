# SIMD — making the model math faster in the browser

## What SIMD is

**SIMD** (*Single Instruction, Multiple Data*) is a CPU instruction that works on several
numbers at once. WebAssembly has **SIMD128**: 128-bit registers that hold 4 `float32` (or 16
`int8`) and multiply-add all of them in one instruction.

```
without SIMD:  y0 += s*w0 → y1 += s*w1 → y2 += s*w2 → y3 += s*w3   (4 instructions)
with SIMD:     [y0 y1 y2 y3] += s * [w0 w1 w2 w3]                     (1 instruction)
```

You meet it here because almost all the time a model spends is multiplying by weight matrices
(`MatmulT` and friends), and that is exactly the kind of loop SIMD speeds up.

This document exists to record **what was measured**, **what it takes to get the speed-up**,
and **how the framework will ship it** without leaving older browsers behind.

## What was measured (2026-09-29)

TinyGo 0.41.1 (LLVM 20), `tinygo test -target … -bench`, on an i7-11800H. The workload is one
generated token through Qwen3.5-0.8B's FFN up projection: a 1 × 1024 vector times a
1024 × 3584 matrix (3.7 M multiply-adds).

| Loop form | `-opt=z` (today's production) | `-opt=2` | `-opt=2` + `simd128` |
|---|---|---|---|
| `MatmulT` as it is (one dot product per output, one accumulator) | 1.31 ms | 1.14 ms | 1.19 ms |
| dot product with 4 accumulators | 1.14 ms | 1.15 ms | 1.12 ms |
| **axpy form** (`y += a[k] · W[k,:]`, weights stored `[k][n]`) | 1.94 ms | 1.37 ms | **0.40 ms** |

**The axpy form with `-opt=2` and `simd128` is 3.3× faster than today's `MatmulT`.** That clears
the adoption threshold of 2×, so it will be adopted.

## Why only that combination works

All three conditions are needed. Missing any one gives no speed-up:

1. **`simd128` must be enabled.** TinyGo's `wasm` target does not enable it. It takes a target
   file that inherits `wasm` and adds `+simd128` to `features` (and `-msimd128` to `cflags`).
2. **The optimization level must be `-opt=2`.** At `-opt=z` (optimize for size, the framework's
   production default) LLVM does not vectorize loops.
3. **The loop must not be a floating-point reduction.** A dot product adds everything into one
   sum. Vectorizing it would change the order of the additions, and therefore the rounding, so
   LLVM refuses. The axpy form updates `n` independent outputs, and each output is still
   summed in the original order. It vectorizes without changing a single result bit per output.

Consequences for the code:
- **One implementation, plain Go.** No assembly, no intrinsics, no build tags. SIMD becomes a
  property of **how the binary is compiled**, not a second copy of the code.
- **Weight layout.** The axpy form reads the matrix row by row as `[k][n]` (input dimension
  outer), the transpose of what `MatmulT` reads today (`bT` as `[n][k]`). The new operation and
  the matching layout in `weights`/`weightsc` (int8 blocks of 32 along `n`) belong to the
  decoder's plans. The encoder keeps `MatmulT` until it is measured the same way.

## How the framework ships it

Today `webtyp/app` produces one artifact, `client.wasm`, compiled for **size**
(`-opt=z -no-debug -panic=trap`). That stays: the page must download fast, and it does no heavy
math.

The model runs in a **Web Worker**, which has its own binary (see the agent's architecture).
Only that binary needs speed, and it is built **twice**:

| Artifact | Flags | Loaded when |
|---|---|---|
| `client.wasm` | `-target wasm -opt=z` (unchanged) | always, by the page |
| `worker.simd.wasm` | `-target wasm-simd.json -opt=2` | the browser supports SIMD128 (Chrome 91+, Firefox 89+, Safari 16.4+) |
| `worker.wasm` | `-target wasm -opt=2` | any other browser: slower, but it works |

The Worker's bootstrap chooses between the last two. It is the one piece of JavaScript the
framework already generates for Workers (`webtyp/js`, `js.WebWorker`), and the choice is a
standard feature test: `WebAssembly.validate(bytes)` on a ~20-byte module containing one SIMD
instruction.

No browser is required to be recent. Old browsers get `worker.wasm` and the same results,
only slower.

Who owns each piece:

| Piece | Repository |
|---|---|
| the axpy operation and its tests | `webtyp/nn` |
| the `wasm-simd.json` target file and the two worker builds | `webtyp/app` (next to its existing size modes) |
| the SIMD feature test in the Worker bootstrap | `webtyp/js` |

## Next measurements

- The same comparison for int8 weights (dequantize inside the axpy loop). WebAssembly has
  `i8 → f32` conversions in SIMD.
- Prefill (many tokens at once) in the axpy form, and the encoder's shapes (`m` > 1).
- Binary size of `-opt=2` versus `-opt=z` for the worker.
