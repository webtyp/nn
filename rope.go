package nn

import (
	"math"

	"webtyp.com/fmt"
)

// RoPE applies Rotary Position Embedding in-place to q and k for position `pos`.
// theta is the rotary base frequency (granite uses 150000/160000 per layer type).
// Rotation is GPT-NeoX style (first/second half), matching rotate_half in
// modeling_modernbert.py — not interleaved pairs.
func RoPE(q, k []float32, pos int, theta float64, dim, heads int) error {
	if dim <= 0 || heads <= 0 || dim%heads != 0 {
		return fmt.Err("nn: invalid dimensions for rope")
	}
	headDim := dim / heads
	if headDim%2 != 0 {
		return fmt.Err("nn: head dimension must be even for rope")
	}

	if len(q) > 0 && len(q) < dim {
		return fmt.Err("nn: q buffer too short for rope")
	}
	if len(k) > 0 && len(k) < dim {
		return fmt.Err("nn: k buffer too short for rope")
	}
	if len(q) == 0 && len(k) == 0 {
		return fmt.Err("nn: no buffers provided for rope")
	}

	rotate := func(v []float32) {
		for h := 0; h < heads; h++ {
			head := v[h*headDim : (h+1)*headDim]
			half := headDim / 2
			for i := 0; i < half; i++ {
				freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
				angle := float64(pos) * freq
				cosA := float32(math.Cos(angle))
				sinA := float32(math.Sin(angle))

				x0 := head[i]
				x1 := head[i+half]
				head[i] = x0*cosA - x1*sinA
				head[i+half] = x0*sinA + x1*cosA
			}
		}
	}

	if len(q) >= dim {
		rotate(q)
	}
	if len(k) >= dim {
		rotate(k)
	}
	return nil
}
