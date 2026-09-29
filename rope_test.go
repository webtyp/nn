package nn

import (
	"math"
	"testing"

	"webtyp.com/vector"
)

func TestRoPE_RotationPreservesNorm(t *testing.T) {
	dim := 64
	heads := 4
	q := make([]float32, dim)
	k := make([]float32, dim)

	for i := range q {
		q[i] = float32(i+1) * 0.1
		k[i] = float32(i+1) * 0.2
	}

	qNormBefore := vector.Norm(q)
	kNormBefore := vector.Norm(k)

	err := RoPE(q, k, 5, 10000.0, dim, heads)
	if err != nil {
		t.Fatalf("RoPE returned error: %v", err)
	}

	qNormAfter := vector.Norm(q)
	kNormAfter := vector.Norm(k)

	if math.Abs(float64(qNormBefore-qNormAfter)) > 1e-5 {
		t.Errorf("q norm before=%v, after=%v", qNormBefore, qNormAfter)
	}
	if math.Abs(float64(kNormBefore-kNormAfter)) > 1e-5 {
		t.Errorf("k norm before=%v, after=%v", kNormBefore, kNormAfter)
	}
}

func TestRoPE_PositionZeroIsIdentity(t *testing.T) {
	dim := 64
	heads := 4
	q := make([]float32, dim)
	qOrig := make([]float32, dim)
	for i := range q {
		q[i] = float32(i+1) * 0.33
		qOrig[i] = q[i]
	}

	err := RoPE(q, nil, 0, 10000.0, dim, heads)
	if err != nil {
		t.Fatalf("RoPE returned error: %v", err)
	}

	for i := range q {
		if math.Abs(float64(q[i]-qOrig[i])) > 1e-6 {
			t.Errorf("At index %d: got %v, expected %v (position 0 modified vector)", i, q[i], qOrig[i])
		}
	}
}
