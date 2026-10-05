package nn

import "webtyp.com/fmt"

// Int4BlockBytes is the size of one 4-bit block of Int8BlockSize values (GGUF Q4_0 layout).
const Int4BlockBytes = Int8BlockSize / 2

// MatVecQ4Block32 computes dst[r] = Σ_c W[r,c]·x[c] with W in 4-bit blocks of 32 (the layout in
// docs/PLAN.md / weights.Int4Block32: byte j of a block holds column j in its low nibble and
// column j+16 in its high nibble, value = (nibble − 8) × scale) and x quantized by
// QuantizeBlocks32. Each block is an int32 dot product scaled once by the two block scales, so
// WebAssembly compilers vectorize it like MatVecQ8Block32.
func MatVecQ4Block32(dst []float32, xq []int8, xs []float32, q []byte, scales []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 || cols%Int8BlockSize != 0 {
		return fmt.Err("nn: MatVecQ4Block32 needs positive rows and cols a multiple of 32")
	}
	blocks := cols / Int8BlockSize
	if len(dst) < rows || len(xq) < cols || len(xs) < blocks || len(q) < rows*cols/2 || len(scales) < rows*blocks {
		return fmt.Err("nn: buffer too short for int4 matvec")
	}

	r := 0
	for ; r+1 < rows; r += 2 {
		w0 := q[r*(cols/2) : (r+1)*(cols/2)]
		w1 := q[(r+1)*(cols/2) : (r+2)*(cols/2)]
		s0 := scales[r*blocks : (r+1)*blocks]
		s1 := scales[(r+1)*blocks : (r+2)*blocks]

		var sum0, sum1 float32
		for b := 0; b < blocks; b++ {
			o := b * Int4BlockBytes
			a0 := w0[o : o+Int4BlockBytes : o+Int4BlockBytes]
			a1 := w1[o : o+Int4BlockBytes : o+Int4BlockBytes]

			k := b * Int8BlockSize
			v := xq[k : k+Int8BlockSize : k+Int8BlockSize]

			var acc0, acc1 int32
			for j := 0; j < Int4BlockBytes; j++ {
				blk0 := int32(a0[j])
				v0 := int32(v[j])
				v16 := int32(v[j+16])
				acc0 += (blk0&0x0F - 8) * v0
				acc0 += (blk0>>4 - 8) * v16

				blk1 := int32(a1[j])
				acc1 += (blk1&0x0F - 8) * v0
				acc1 += (blk1>>4 - 8) * v16
			}
			sum0 += float32(acc0) * (s0[b] * xs[b])
			sum1 += float32(acc1) * (s1[b] * xs[b])
		}
		dst[r], dst[r+1] = sum0, sum1
	}

	if r < rows {
		w := q[r*(cols/2) : (r+1)*(cols/2)]
		s := scales[r*blocks : (r+1)*blocks]

		var sum float32
		for b := 0; b < blocks; b++ {
			o := b * Int4BlockBytes
			a := w[o : o+Int4BlockBytes : o+Int4BlockBytes]

			k := b * Int8BlockSize
			v := xq[k : k+Int8BlockSize : k+Int8BlockSize]

			var acc int32
			for j := 0; j < Int4BlockBytes; j++ {
				blk := int32(a[j])
				acc += (blk&0x0F - 8) * int32(v[j])
				acc += (blk>>4 - 8) * int32(v[j+16])
			}
			sum += float32(acc) * (s[b] * xs[b])
		}
		dst[r] = sum
	}

	return nil
}
