package nn

import (
	"math"

	"webtyp.com/fmt"
)

// LayerNorm normalizes src over the last dimension `dim` with gamma and beta scaling.
// Mean and variance are accumulated in float64 to prevent drift.
func LayerNorm(dst, src, gamma, beta []float32, dim int, eps float32) error {
	if dim <= 0 {
		return fmt.Err("nn: invalid dimension for layernorm")
	}
	if len(src)%dim != 0 || len(dst) != len(src) {
		return fmt.Err("nn: buffer size mismatch for layernorm")
	}
	if len(gamma) > 0 && len(gamma) < dim {
		return fmt.Err("nn: gamma buffer too short for layernorm")
	}
	if len(beta) > 0 && len(beta) < dim {
		return fmt.Err("nn: beta buffer too short for layernorm")
	}

	numRows := len(src) / dim
	for r := 0; r < numRows; r++ {
		off := r * dim
		sRow := src[off : off+dim]
		dRow := dst[off : off+dim]

		var sum float64
		for _, v := range sRow {
			sum += float64(v)
		}
		mean := sum / float64(dim)

		var varSum float64
		for _, v := range sRow {
			diff := float64(v) - mean
			varSum += diff * diff
		}
		variance := varSum / float64(dim)
		invStd := 1.0 / math.Sqrt(variance+float64(eps))

		for j := 0; j < dim; j++ {
			xNorm := (float64(sRow[j]) - mean) * invStd
			g := 1.0
			if len(gamma) > 0 {
				g = float64(gamma[j])
			}
			b := 0.0
			if len(beta) > 0 {
				b = float64(beta[j])
			}
			dRow[j] = float32(xNorm*g + b)
		}
	}
	return nil
}

// RMSNorm normalizes src over the last dimension `dim` using root-mean-square.
// Sum of squares is accumulated in float64 to prevent precision loss.
func RMSNorm(dst, src, gamma []float32, dim int, eps float32) error {
	if dim <= 0 {
		return fmt.Err("nn: invalid dimension for rmsnorm")
	}
	if len(src)%dim != 0 || len(dst) != len(src) {
		return fmt.Err("nn: buffer size mismatch for rmsnorm")
	}
	if len(gamma) > 0 && len(gamma) < dim {
		return fmt.Err("nn: gamma buffer too short for rmsnorm")
	}

	numRows := len(src) / dim
	for r := 0; r < numRows; r++ {
		off := r * dim
		sRow := src[off : off+dim]
		dRow := dst[off : off+dim]

		var sqSum float64
		for _, v := range sRow {
			vf := float64(v)
			sqSum += vf * vf
		}
		rms := 1.0 / math.Sqrt(sqSum/float64(dim)+float64(eps))

		for j := 0; j < dim; j++ {
			g := 1.0
			if len(gamma) > 0 {
				g = float64(gamma[j])
			}
			dRow[j] = float32(float64(sRow[j]) * rms * g)
		}
	}
	return nil
}
