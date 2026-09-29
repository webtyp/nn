package nn

import (
	"math"
	"testing"
)

func TestLayerNorm_ZeroMeanUnitVar(t *testing.T) {
	dim := 64
	src := make([]float32, dim)
	for i := range src {
		src[i] = float32(i)*0.5 - 10.0
	}
	dst := make([]float32, dim)

	// No gamma/beta scaling
	err := LayerNorm(dst, src, nil, nil, dim, 1e-5)
	if err != nil {
		t.Fatalf("LayerNorm returned error: %v", err)
	}

	var sum float64
	for _, v := range dst {
		sum += float64(v)
	}
	mean := sum / float64(dim)

	var varSum float64
	for _, v := range dst {
		diff := float64(v) - mean
		varSum += diff * diff
	}
	variance := varSum / float64(dim)

	if math.Abs(mean) > 1e-5 {
		t.Errorf("LayerNorm mean = %v, expected ~0", mean)
	}
	if math.Abs(variance-1.0) > 1e-4 {
		t.Errorf("LayerNorm variance = %v, expected ~1", variance)
	}
}

func TestLayerNorm_Float64Accumulation(t *testing.T) {
	dim := 384
	src := make([]float32, dim)
	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	for i := range src {
		src[i] = 1000.0 + float32(i)*0.01
		gamma[i] = 1.0
		beta[i] = 0.0
	}
	dst := make([]float32, dim)

	err := LayerNorm(dst, src, gamma, beta, dim, 1e-5)
	if err != nil {
		t.Fatalf("LayerNorm returned error: %v", err)
	}

	// Compute float64 reference
	var sum float64
	for _, v := range src {
		sum += float64(v)
	}
	mean64 := sum / float64(dim)

	var varSum float64
	for _, v := range src {
		diff := float64(v) - mean64
		varSum += diff * diff
	}
	variance64 := varSum / float64(dim)
	invStd64 := 1.0 / math.Sqrt(variance64+1e-5)

	for i := 0; i < dim; i++ {
		ref := float32((float64(src[i]) - mean64) * invStd64)
		diff := math.Abs(float64(dst[i] - ref))
		if diff > 1e-6 {
			t.Errorf("At index %d: dst=%v, ref=%v (diff %v)", i, dst[i], ref, diff)
		}
	}
}
