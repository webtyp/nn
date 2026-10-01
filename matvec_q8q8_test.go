package nn

import (
	"math"
	"testing"
)

func q8Data(rows, cols int) ([]float32, []byte, []float32) {
	q := make([]byte, rows*cols)
	seed := uint32(7)
	for i := range q {
		seed = seed*1664525 + 1013904223
		q[i] = byte(int8(int32(seed>>24) - 128))
	}
	s := make([]float32, rows*cols/Int8BlockSize)
	for i := range s {
		s[i] = 0.002 + float32(i%5)*0.0004
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.37)) * 0.8
	}
	return x, q, s
}

// The int8×int8 kernel stays within the activation quantization error of the float kernel,
// for an odd number of rows (the last row runs alone).
func TestMatVecQ8Block32_CloseToFloatKernel(t *testing.T) {
	rows, cols := 33, 256
	x, q, s := q8Data(rows, cols)
	want := make([]float32, rows)
	if err := MatVecInt8Block32(want, x, q, s, rows, cols); err != nil {
		t.Fatal(err)
	}
	xq, xs := make([]int8, cols), make([]float32, cols/Int8BlockSize)
	if err := QuantizeBlocks32(xq, xs, x); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, rows)
	if err := MatVecQ8Block32(got, xq, xs, q, s, rows, cols); err != nil {
		t.Fatal(err)
	}
	var num, den float64
	for i := range got {
		e := float64(got[i] - want[i])
		num += e * e
		den += float64(want[i]) * float64(want[i])
	}
	if rel := math.Sqrt(num / den); rel > 0.01 {
		t.Fatalf("relative RMS error %.4f, want below 1%%", rel)
	}
}

func TestQuantizeBlocks32_RoundTrip(t *testing.T) {
	x := []float32{0, 0.5, -1, 0.25}
	for len(x) < 32 {
		x = append(x, 0)
	}
	xq, xs := make([]int8, 32), make([]float32, 1)
	if err := QuantizeBlocks32(xq, xs, x); err != nil {
		t.Fatal(err)
	}
	if xq[2] != -127 || xs[0] != float32(1)/127 {
		t.Fatalf("amax maps to ±127: got q=%d scale=%v", xq[2], xs[0])
	}
	for i, v := range x[:4] {
		if d := float32(xq[i])*xs[0] - v; d > xs[0]/2+1e-7 || d < -xs[0]/2-1e-7 {
			t.Fatalf("value %d: %v decoded as %v", i, v, float32(xq[i])*xs[0])
		}
	}
}

func TestMatVecQ8Block32_RejectsBadShapes(t *testing.T) {
	if err := MatVecQ8Block32(make([]float32, 1), make([]int8, 40), make([]float32, 2), make([]byte, 40), make([]float32, 2), 1, 40); err == nil {
		t.Fatal("cols must be a multiple of 32")
	}
}

// Under TinyGo the integer kernel vectorizes; run both with:
//   tinygo test -target wasm -opt=2 -bench MatVec -run XXX .
//   tinygo test -target testdata/wasm-simd.json -opt=2 -bench MatVec -run XXX .
func BenchmarkMatVecInt8Block32(b *testing.B) {
	x, q, s := q8Data(3584, 1024)
	d := make([]float32, 3584)
	for n := 0; n < b.N; n++ {
		_ = MatVecInt8Block32(d, x, q, s, 3584, 1024)
	}
}

func BenchmarkMatVecQ8Block32(b *testing.B) {
	x, q, s := q8Data(3584, 1024)
	d := make([]float32, 3584)
	xq, xs := make([]int8, 1024), make([]float32, 32)
	for n := 0; n < b.N; n++ {
		_ = QuantizeBlocks32(xq, xs, x)
		_ = MatVecQ8Block32(d, xq, xs, q, s, 3584, 1024)
	}
}
