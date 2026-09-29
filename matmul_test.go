package nn

import (
	"math"
	"testing"
)

func TestMatmulT_MatchesNaive(t *testing.T) {
	m, k, n := 4, 8, 5
	a := make([]float32, m*k)
	bT := make([]float32, n*k)
	for i := range a {
		a[i] = float32(i+1) * 0.1
	}
	for i := range bT {
		bT[i] = float32(i+1) * 0.05
	}

	dst := make([]float32, m*n)
	err := MatmulT(dst, a, bT, m, k, n)
	if err != nil {
		t.Fatalf("MatmulT returned unexpected error: %v", err)
	}

	// Naive reference calculation
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var expected float32
			for p := 0; p < k; p++ {
				expected += a[i*k+p] * bT[j*k+p]
			}
			actual := dst[i*n+j]
			diff := math.Abs(float64(actual - expected))
			if diff > 1e-5*math.Max(1.0, float64(math.Abs(float64(expected)))) {
				t.Errorf("At dst[%d,%d]: got %v, expected %v (diff %v)", i, j, actual, expected, diff)
			}
		}
	}
}

func TestMatmulT_Dimensions(t *testing.T) {
	dst := make([]float32, 10)
	a := make([]float32, 10)
	bT := make([]float32, 10)

	// Dimension mismatch (dst too short for 4x4=16)
	err := MatmulT(dst, a, bT, 4, 2, 4)
	if err == nil {
		t.Errorf("Expected error for buffer too short, got nil")
	}

	// Invalid zero/negative dimensions
	err = MatmulT(dst, a, bT, 0, 2, 2)
	if err == nil {
		t.Errorf("Expected error for m <= 0, got nil")
	}
}
