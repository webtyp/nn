package nn

import (
	"math"
	"testing"
)

func TestGELU_KnownValues(t *testing.T) {
	// GELU(0) = 0
	// GELU(large negative) = ~0
	// GELU(large positive) = ~x
	x := []float32{0.0, -10.0, 10.0, 1.0}
	err := GELU(x)
	if err != nil {
		t.Fatalf("GELU returned error: %v", err)
	}

	if math.Abs(float64(x[0])) > 1e-6 {
		t.Errorf("GELU(0) = %v, expected 0", x[0])
	}
	if math.Abs(float64(x[1])) > 1e-5 {
		t.Errorf("GELU(-10) = %v, expected ~0", x[1])
	}
	if math.Abs(float64(x[2]-10.0)) > 1e-5 {
		t.Errorf("GELU(10) = %v, expected ~10", x[2])
	}

	// GELU(1) = 1 * Phi(1) = 1 * 0.8413447 = ~0.8413447
	expected1 := float32(0.8413447)
	if math.Abs(float64(x[3]-expected1)) > 1e-5 {
		t.Errorf("GELU(1) = %v, expected ~%v", x[3], expected1)
	}
}

func TestSiLU_KnownValues(t *testing.T) {
	// SiLU(0) = 0
	// SiLU(-10) = -10 / (1 + e^10) ~ -0.000454
	// SiLU(10) = 10 / (1 + e^-10) ~ 9.999546
	x := []float32{0.0, -10.0, 10.0, 1.0}
	err := SiLU(x)
	if err != nil {
		t.Fatalf("SiLU returned error: %v", err)
	}

	if math.Abs(float64(x[0])) > 1e-6 {
		t.Errorf("SiLU(0) = %v, expected 0", x[0])
	}
	if math.Abs(float64(x[1]-(-0.0004539787))) > 1e-5 {
		t.Errorf("SiLU(-10) = %v, expected ~ -0.0004539787", x[1])
	}
	if math.Abs(float64(x[2]-9.999546)) > 1e-5 {
		t.Errorf("SiLU(10) = %v, expected ~9.999546", x[2])
	}

	// SiLU(1) = 1 / (1 + e^-1) = 1 / (1 + 0.36787944) = ~0.7310586
	expected1 := float32(0.7310586)
	if math.Abs(float64(x[3]-expected1)) > 1e-5 {
		t.Errorf("SiLU(1) = %v, expected ~%v", x[3], expected1)
	}
}
