package nn

import (
	"math"
	"testing"
)

func q4Data(rows, cols int) ([]float32, []byte, []float32) {
	q := make([]byte, rows*cols/2)
	seed := uint32(7)
	for i := range q {
		seed = seed*1664525 + 1013904223
		q[i] = byte(seed & 0xFF)
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

func dequant4(q []byte, scales []float32, rows, cols int) []float32 {
	dst := make([]float32, rows*cols)
	blocks := cols / Int8BlockSize
	for r := 0; r < rows; r++ {
		for b := 0; b < blocks; b++ {
			o := (r*blocks + b) * Int4BlockBytes
			s := scales[r*blocks + b]
			for j := 0; j < Int4BlockBytes; j++ {
				blk := q[o+j]
				dst[r*cols + b*Int8BlockSize + j] = float32(int32(blk&0x0F) - 8) * s
				dst[r*cols + b*Int8BlockSize + j + 16] = float32(int32(blk>>4) - 8) * s
			}
		}
	}
	return dst
}

func TestMatVecQ4Block32_MatchesDequantized(t *testing.T) {
	rows, cols := 33, 256
	x, q, s := q4Data(rows, cols)

	xq, xs := make([]int8, cols), make([]float32, cols/Int8BlockSize)
	if err := QuantizeBlocks32(xq, xs, x); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, rows)
	if err := MatVecQ4Block32(got, xq, xs, q, s, rows, cols); err != nil {
		t.Fatal(err)
	}

	want := make([]float32, rows)
	w := dequant4(q, s, rows, cols)
	for r := 0; r < rows; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += w[r*cols+c] * (float32(xq[c]) * xs[c/Int8BlockSize])
		}
		want[r] = sum
	}

	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-4*math.Abs(float64(want[i])) && math.Abs(float64(got[i]-want[i])) > 1e-7 {
			t.Fatalf("row %d: got %v, want %v (rel error > 1e-4)", i, got[i], want[i])
		}
	}
}

func TestMatVecQ4Block32_CloseToFloatX(t *testing.T) {
	rows, cols := 33, 256
	x, q, s := q4Data(rows, cols)

	xq, xs := make([]int8, cols), make([]float32, cols/Int8BlockSize)
	if err := QuantizeBlocks32(xq, xs, x); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, rows)
	if err := MatVecQ4Block32(got, xq, xs, q, s, rows, cols); err != nil {
		t.Fatal(err)
	}

	want := make([]float32, rows)
	w := dequant4(q, s, rows, cols)
	for r := 0; r < rows; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += w[r*cols+c] * x[c]
		}
		want[r] = sum
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

func TestMatVecQ4Block32_NibbleOrder(t *testing.T) {
	rows, cols := 1, 32
	q := make([]byte, cols/2)
	s := make([]float32, 1)
	x := make([]float32, cols)

	q[0] = 0x9F
	for i := 1; i < len(q); i++ {
		q[i] = 0x88
	}
	s[0] = 1.0
	x[0] = 1.0
	x[16] = 2.0

	xq, xs := make([]int8, cols), make([]float32, cols/Int8BlockSize)
	if err := QuantizeBlocks32(xq, xs, x); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, rows)
	if err := MatVecQ4Block32(got, xq, xs, q, s, rows, cols); err != nil {
		t.Fatal(err)
	}

	// float32(xq[0]) * xs[0] should be approx 1.0
	// float32(xq[16]) * xs[0] should be approx 2.0
	// low nibble 0xF -> 15-8 = 7
	// high nibble 0x9 -> 9-8 = 1
	// 7 * 1 + 1 * 2 = 9
	if math.Abs(float64(got[0] - 9.0)) > 0.1 {
		t.Fatalf("got %v, want approx 9.0", got[0])
	}
}

func TestMatVecQ4Block32_RejectsBadShapes(t *testing.T) {
	if err := MatVecQ4Block32(make([]float32, 1), make([]int8, 40), make([]float32, 2), make([]byte, 20), make([]float32, 2), 1, 40); err == nil {
		t.Fatal("cols must be a multiple of 32")
	}
	if err := MatVecQ4Block32(make([]float32, 1), make([]int8, 32), make([]float32, 1), make([]byte, 15), make([]float32, 1), 1, 32); err == nil {
		t.Fatal("q one byte short")
	}
}

func BenchmarkMatVecQ4Block32(b *testing.B) {
	x, q, s := q4Data(3584, 1024)
	d := make([]float32, 3584)
	xq, xs := make([]int8, 1024), make([]float32, 32)
	for n := 0; n < b.N; n++ {
		_ = QuantizeBlocks32(xq, xs, x)
		_ = MatVecQ4Block32(d, xq, xs, q, s, 3584, 1024)
	}
}
