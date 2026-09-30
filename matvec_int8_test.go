package nn

import (
	"math"
	"testing"
)

// The int8 kernel equals dequantizing each value (q × scale of its block) and multiplying in
// float32, including a last block shorter than Int8BlockSize.
func TestMatVecInt8Block32_MatchesDequantized(t *testing.T) {
	rows, cols := 3, 70 // 3 blocks per row, the last one 6 wide
	blocks := 3
	q := make([]byte, rows*cols)
	scales := make([]float32, rows*blocks)
	x := make([]float32, cols)
	for i := range q {
		q[i] = byte(int8((i*37)%255 - 127))
	}
	for i := range scales {
		scales[i] = 0.01 * float32(i+1)
	}
	for i := range x {
		x[i] = float32(math.Sin(float64(i)))
	}

	got := make([]float32, rows)
	if err := MatVecInt8Block32(got, x, q, scales, rows, cols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < rows; r++ {
		var want float64
		for c := 0; c < cols; c++ {
			w := float64(int8(q[r*cols+c])) * float64(scales[r*blocks+c/Int8BlockSize])
			want += w * float64(x[c])
		}
		if math.Abs(float64(got[r])-want) > 1e-4 {
			t.Errorf("row %d: got %v, want %v", r, got[r], want)
		}
	}
}

func TestMatVecInt8Block32_RejectsShortBuffers(t *testing.T) {
	if err := MatVecInt8Block32(make([]float32, 2), make([]float32, 32), make([]byte, 64), make([]float32, 1), 2, 32); err == nil {
		t.Fatal("expected an error: 2 rows need 2 scales")
	}
	if err := MatVecInt8Block32(nil, nil, nil, nil, 0, 32); err == nil {
		t.Fatal("expected an error for zero rows")
	}
}

// Multiplying m inputs at once equals m separate matrix-vector products.
func TestMatmulInt8Block32_EqualsRepeatedMatVec(t *testing.T) {
	m, rows, cols := 4, 3, 70
	blocks := 3
	q := make([]byte, rows*cols)
	scales := make([]float32, rows*blocks)
	x := make([]float32, m*cols)
	for i := range q {
		q[i] = byte(int8((i*53)%255 - 127))
	}
	for i := range scales {
		scales[i] = 0.02 * float32(i+1)
	}
	for i := range x {
		x[i] = float32(math.Cos(float64(i) * 0.3))
	}
	got := make([]float32, m*rows)
	if err := MatmulInt8Block32(got, x, q, scales, m, rows, cols); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, rows)
	for i := 0; i < m; i++ {
		if err := MatVecInt8Block32(want, x[i*cols:(i+1)*cols], q, scales, rows, cols); err != nil {
			t.Fatal(err)
		}
		for r := 0; r < rows; r++ {
			if got[i*rows+r] != want[r] {
				t.Errorf("input %d row %d: got %v, want %v", i, r, got[i*rows+r], want[r])
			}
		}
	}
}
