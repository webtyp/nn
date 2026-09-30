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
		row := q[r*cols : (r+1)*cols]
		rowScales := scales[r*blocks : (r+1)*blocks]
		var sum float32
		for b := 0; b < blocks; b++ {
			start := b * Int8BlockSize
			end := start + Int8BlockSize
			if end > cols {
				end = cols
			}
			var acc float32
			for c := start; c < end; c++ {
				acc += float32(int8(row[c])) * x[c]
			}
			sum += acc * rowScales[b]
		}
		dst[r] = sum
	}
	return nil
}
