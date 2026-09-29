package nn

import (
	"math"
)

// GELU applies GELU activation function in-place over `x`.
func GELU(x []float32) error {
	for i, v := range x {
		vf := float64(v)
		x[i] = float32(0.5 * vf * (1.0 + math.Erf(vf/math.Sqrt2)))
	}
	return nil
}

// SiLU applies SiLU (Swish-1) activation function in-place over `x`.
func SiLU(x []float32) error {
	for i, v := range x {
		vf := float64(v)
		x[i] = float32(vf / (1.0 + math.Exp(-vf)))
	}
	return nil
}
