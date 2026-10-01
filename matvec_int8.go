package nn

import "webtyp.com/fmt"

// Int8BlockSize is how many consecutive values of a row share one scale in the int8 block
// layout (GGUF Q8_0, webtyp/weights Int8Block32).
const Int8BlockSize = 32

// MatVecInt8Block32 computes dst[r] = Σ_c W[r,c]·x[c] for a rows×cols matrix W stored as int8
// values in blocks: q holds rows×cols bytes, row-major, each byte an int8 (two's complement), and
// scales holds rows×ceil(cols/Int8BlockSize) entries, row by row, with W[r,c] = q × scale of its
// block. The weights stay int8 in memory: a model keeps a quarter of the float32 size.
func MatVecInt8Block32(dst, x []float32, q []byte, scales []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Err("nn: invalid matvec dimensions")
	}
	blocks := (cols + Int8BlockSize - 1) / Int8BlockSize
	if len(dst) < rows || len(x) < cols || len(q) < rows*cols || len(scales) < rows*blocks {
		return fmt.Err("nn: buffer too short for int8 matvec")
	}
	for r := 0; r < rows; r++ {
		dst[r] = dotInt8Block32(q[r*cols:(r+1)*cols], scales[r*blocks:(r+1)*blocks], x[:cols])
	}
	return nil
}

// MatmulInt8Block32 computes dst[i,r] = Σ_c W[r,c]·x[i,c] for m inputs at once (dst is m×rows,
// x is m×cols), with W stored as in MatVecInt8Block32. Each row of W is read once for all m
// inputs, which is what makes reading a prompt in one pass faster than token by token.
func MatmulInt8Block32(dst, x []float32, q []byte, scales []float32, m, rows, cols int) error {
	if m <= 0 || rows <= 0 || cols <= 0 {
		return fmt.Err("nn: invalid matmul dimensions")
	}
	blocks := (cols + Int8BlockSize - 1) / Int8BlockSize
	if len(dst) < m*rows || len(x) < m*cols || len(q) < rows*cols || len(scales) < rows*blocks {
		return fmt.Err("nn: buffer too short for int8 matmul")
	}
	for r := 0; r < rows; r++ {
		row := q[r*cols : (r+1)*cols]
		rowScales := scales[r*blocks : (r+1)*blocks]
		for i := 0; i < m; i++ {
			dst[i*rows+r] = dotInt8Block32(row, rowScales, x[i*cols:(i+1)*cols])
		}
	}
	return nil
}

// dotInt8Block32 is Σ_c W[c]·x[c] for one row of int8 blocks. Full blocks run eight independent
// accumulators over re-sliced, fixed-length views, so the compiler drops the bounds checks and
// the additions do not wait on each other: measured 1.4× faster than one accumulator.
func dotInt8Block32(row []byte, scales []float32, x []float32) float32 {
	cols := len(x)
	full := cols / Int8BlockSize
	var sum float32
	for b := 0; b < full; b++ {
		w := row[b*Int8BlockSize : b*Int8BlockSize+Int8BlockSize : b*Int8BlockSize+Int8BlockSize]
		xv := x[b*Int8BlockSize : b*Int8BlockSize+Int8BlockSize : b*Int8BlockSize+Int8BlockSize]
		var a0, a1, a2, a3, a4, a5, a6, a7 float32
		for c := 0; c < Int8BlockSize; c += 8 {
			a0 += float32(int8(w[c])) * xv[c]
			a1 += float32(int8(w[c+1])) * xv[c+1]
			a2 += float32(int8(w[c+2])) * xv[c+2]
			a3 += float32(int8(w[c+3])) * xv[c+3]
			a4 += float32(int8(w[c+4])) * xv[c+4]
			a5 += float32(int8(w[c+5])) * xv[c+5]
			a6 += float32(int8(w[c+6])) * xv[c+6]
			a7 += float32(int8(w[c+7])) * xv[c+7]
		}
		sum += ((a0 + a1) + (a2 + a3) + ((a4 + a5) + (a6 + a7))) * scales[b]
	}
	if full*Int8BlockSize < cols {
		var acc float32
		for c := full * Int8BlockSize; c < cols; c++ {
			acc += float32(int8(row[c])) * x[c]
		}
		sum += acc * scales[full]
	}
	return sum
}
