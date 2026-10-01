package nn

import "webtyp.com/fmt"

// QuantizeBlocks32 quantizes x to int8 in blocks of Int8BlockSize, with one scale per block
// (amax/127, rounded to nearest): xq has len(x) values and xs ceil(len(x)/32) scales. A model
// quantizes each input vector once, then multiplies it by every int8 matrix with
// MatVecQ8Block32.
func QuantizeBlocks32(xq []int8, xs []float32, x []float32) error {
	blocks := (len(x) + Int8BlockSize - 1) / Int8BlockSize
	if len(xq) < len(x) || len(xs) < blocks {
		return fmt.Err("nn: buffer too short for block quantization")
	}
	for b := 0; b < blocks; b++ {
		start := b * Int8BlockSize
		end := start + Int8BlockSize
		if end > len(x) {
			end = len(x)
		}
		var amax float32
		for _, f := range x[start:end] {
			if f < 0 {
				f = -f
			}
			if f > amax {
				amax = f
			}
		}
		scale := amax / 127
		xs[b] = scale
		var inv float32
		if scale != 0 {
			inv = 1 / scale
		}
		for i, f := range x[start:end] {
			t := f * inv
			if t >= 0 {
				t += 0.5
			} else {
				t -= 0.5
			}
			xq[start+i] = int8(t)
		}
	}
	return nil
}

// MatVecQ8Block32 computes dst[r] = Σ_c W[r,c]·x[c] with both W and x in int8 blocks of 32
// (W as in MatVecInt8Block32, x from QuantizeBlocks32). Each block is an int32 dot product,
// scaled once by the two block scales. The integer reduction is what WebAssembly compilers
// vectorize (SIMD128), which a float reduction cannot be: measured under TinyGo 0.41 on a
// 3584×1024 matrix, 1.30 ms without SIMD and 0.61 ms with it, against 2.6 ms for the float
// kernel (nn/docs/PERFORMANCE.md). Its result differs from the float kernel by the activation
// quantization, about 0.3 % relative RMS.
func MatVecQ8Block32(dst []float32, xq []int8, xs []float32, q []byte, scales []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 || cols%Int8BlockSize != 0 {
		return fmt.Err("nn: MatVecQ8Block32 needs positive rows and cols a multiple of 32")
	}
	blocks := cols / Int8BlockSize
	if len(dst) < rows || len(xq) < cols || len(xs) < blocks || len(q) < rows*cols || len(scales) < rows*blocks {
		return fmt.Err("nn: buffer too short for int8 matvec")
	}
	r := 0
	// Two rows at a time: each activation block is loaded once for both.
	for ; r+1 < rows; r += 2 {
		w0 := q[r*cols : (r+1)*cols]
		w1 := q[(r+1)*cols : (r+2)*cols]
		s0 := scales[r*blocks : (r+1)*blocks]
		s1 := scales[(r+1)*blocks : (r+2)*blocks]
		var sum0, sum1 float32
		for b := 0; b < blocks; b++ {
			o := b * Int8BlockSize
			a := w0[o : o+Int8BlockSize : o+Int8BlockSize]
			c := w1[o : o+Int8BlockSize : o+Int8BlockSize]
			v := xq[o : o+Int8BlockSize : o+Int8BlockSize]
			var acc0, acc1 int32
			for i := 0; i < Int8BlockSize; i++ {
				x := int32(v[i])
				acc0 += int32(int8(a[i])) * x
				acc1 += int32(int8(c[i])) * x
			}
			sum0 += float32(acc0) * (s0[b] * xs[b])
			sum1 += float32(acc1) * (s1[b] * xs[b])
		}
		dst[r], dst[r+1] = sum0, sum1
	}
	if r < rows {
		w := q[r*cols : (r+1)*cols]
		s := scales[r*blocks : (r+1)*blocks]
		var sum float32
		for b := 0; b < blocks; b++ {
			o := b * Int8BlockSize
			a := w[o : o+Int8BlockSize : o+Int8BlockSize]
			v := xq[o : o+Int8BlockSize : o+Int8BlockSize]
			var acc int32
			for i := 0; i < Int8BlockSize; i++ {
				acc += int32(int8(a[i])) * int32(v[i])
			}
			sum += float32(acc) * (s[b] * xs[b])
		}
		dst[r] = sum
	}
	return nil
}
