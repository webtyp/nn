---
PLAN: "feat: MatVecQ4Block32 — 4-bit weights × int8 activations, the integer kernel for Q4_0 matrices"
TAG: v0.5.0
EXECUTOR: jules
REVIEWER: none
STATUS: running
SESSION: 4088408744144740277
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.

# Plan — `nn` v0.5.0: `MatVecQ4Block32`

**Read [AGENTS.md](../AGENTS.md) and [docs/SIMD.md](SIMD.md) first**: this is a browser library,
`gotest -tinygo` decides, and SIMD only vectorizes **integer** loops (no float reduction inside the
block). Master plan:
[AGENT_ECOSYSTEM_MASTER_PLAN.md](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md)
(D13: 4-bit blocks). Background: [docs/PERFORMANCE.md](PERFORMANCE.md).

## Why

Model matrices will also be stored in 4 bits (`webtyp.com/weights` `Int4Block32`, being added in
parallel: GGUF `Q4_0` layout with float32 scales). The decoder needs the same kind of kernel it has
for int8 (`MatVecQ8Block32`): the input vector quantized once to int8 blocks
(`QuantizeBlocks32`, existing), then each block an **integer** dot product scaled once by the two
block scales. **This repo does not import `weights`**: the layout is restated here.

## The layout (input of the new kernel)

`q` is a rows×cols matrix (cols a multiple of 32), row-major. Each row is `cols/32` blocks of
**16 bytes**. In block `b` of row `r`, byte `j` (0..15) holds the value of column `b*32 + j` in its
**low nibble** and the value of column `b*32 + j + 16` in its **high nibble**. A nibble `n` stands
for `(n − 8) × scales[r*(cols/32) + b]`. So `len(q) == rows*cols/2` and
`len(scales) == rows*cols/32`.

## Design gate

1. **Prior art.** llama.cpp `ggml_vec_dot_q4_0_q8_0`: unpack the two nibbles of each byte, subtract
   8, integer dot product with the int8 activation block, scale by `d_w × d_x`. Its WASM SIMD path
   does the unpacking with byte masks and shifts — integer ops LLVM also emits from plain Go.
2. **Novice-name test.** `MatVecQ4Block32` beside `MatVecQ8Block32`; `Int4BlockBytes = 16`.
3. **Complexity ledger.** +1 kernel, +1 constant. Same arguments as `MatVecQ8Block32`.
4. **Where it belongs.** Here, with the other kernels.
5. **What it deletes.** Nothing.

## Stage 1 — `matvec_q4q8.go` (new)

```go
// Int4BlockBytes is the size of one 4-bit block of Int8BlockSize values (GGUF Q4_0 layout).
const Int4BlockBytes = Int8BlockSize / 2

// MatVecQ4Block32 computes dst[r] = Σ_c W[r,c]·x[c] with W in 4-bit blocks of 32 (the layout in
// docs/PLAN.md / weights.Int4Block32: byte j of a block holds column j in its low nibble and
// column j+16 in its high nibble, value = (nibble − 8) × scale) and x quantized by
// QuantizeBlocks32. Each block is an int32 dot product scaled once by the two block scales, so
// WebAssembly compilers vectorize it like MatVecQ8Block32.
func MatVecQ4Block32(dst []float32, xq []int8, xs []float32, q []byte, scales []float32, rows, cols int) error
```

- Errors (exact text, as `fmt.Err`): rows ≤ 0, cols ≤ 0 or `cols%32 != 0` →
  `nn: MatVecQ4Block32 needs positive rows and cols a multiple of 32`; any slice too short
  (`len(dst) < rows`, `len(xq) < cols`, `len(xs) < cols/32`, `len(q) < rows*cols/2`,
  `len(scales) < rows*cols/32`) → `nn: buffer too short for int4 matvec`.
- **Two rows at a time**, like `MatVecQ8Block32` (each activation block loaded once for both rows),
  then the last row alone when `rows` is odd.
- Inner loop per block, for each of the two rows, with fixed-length re-slices
  (`w[o : o+16 : o+16]`, `xq[k : k+32 : k+32]`) so bounds checks disappear:

  ```go
  var acc int32
  for j := 0; j < Int4BlockBytes; j++ {
  	b := int32(blk[j])
  	acc += (b&0x0F - 8) * int32(v[j])
  	acc += (b>>4 - 8) * int32(v[j+16])
  }
  sum += float32(acc) * (s[blkIdx] * xs[blkIdx])
  ```

  No float arithmetic inside the `j` loop. Keep `- 8` per nibble (do not factor out
  `8·Σx` unless the benchmark below proves it faster under TinyGo SIMD; if it does, document both
  numbers).

## Stage 2 — tests (`matvec_q4q8_test.go`, root, `package nn`, like `matvec_q8q8_test.go`)

A helper `q4Data(rows, cols)` builds deterministic nibbles (LCG as in `q8Data`, `& 0x0F` per
nibble), scales and an `x`, and a reference `dequant4(q, scales, rows, cols) []float32` that
unpacks exactly per the layout above (written independently, plain loops).

| Test | Proves |
|---|---|
| `TestMatVecQ4Block32_MatchesDequantized` | rows 33 (odd), cols 256: result vs the float reference `Σ dequant4(W)[r,c] · (xq[c]·xs[c/32])` within 1e-4 relative (same quantized x, so only float summation order differs) |
| `TestMatVecQ4Block32_CloseToFloatX` | vs `Σ dequant4(W)·x` with the unquantized x: relative RMS below 1 % (activation quantization only) |
| `TestMatVecQ4Block32_NibbleOrder` | one row, one block: byte 0 = `0x9F` (low 15 → +7, high 9 → +1), all other bytes `0x88` (zeros), scale 1; x = 1 at column 0 and 2 at column 16, quantized → dst within 0.1 of 7·1 + 1·2 = 9 |
| `TestMatVecQ4Block32_RejectsBadShapes` | cols 40 → the shape error; `q` one byte short → the buffer error |
| `BenchmarkMatVecQ4Block32` | 3584×1024, like the existing Q8 benchmark |

## Stage 3 — measure and document

Run, and paste the numbers into `docs/PERFORMANCE.md` under a new heading
"4-bit weights (`MatVecQ4Block32`)" as a table next to the int8 one (WebAssembly no SIMD, WebAssembly
+ SIMD128, native Go), with the commands:

```bash
tinygo test -target wasm -opt=2 -bench 'MatVecQ[48]Block32' -run XXX .
tinygo test -target testdata/wasm-simd.json -opt=2 -bench 'MatVecQ[48]Block32' -run XXX .
go test -bench 'MatVecQ[48]Block32' -run XXX .
```

If a command cannot run in the executor's environment, write "not measured (no TinyGo/Node in the
executor)" in that cell — never an estimate.

`README.md`: the kernel table gets `MatVecQ4Block32`.

## Acceptance

- `gotest` and `gotest -tinygo` green. Never run `gopush` or `codejob`.
- `grep -n "float32(" matvec_q4q8.go` shows no conversion inside the `j` loop.

| Stage | Files | Done when |
|---|---|---|
| 1 | `matvec_q4q8.go` | kernel |
| 2 | `matvec_q4q8_test.go` | table green |
| 3 | `docs/PERFORMANCE.md`, `README.md` | measured or marked not measured |

## Executor notes
- WebAssembly performance was not measured using `tinygo test` because TinyGo was not available in the executor's environment. I marked these values as "not measured (no TinyGo/Node in the executor)" in `docs/PERFORMANCE.md` as instructed.
