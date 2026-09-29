package nn

import (
	"math"
)

// Softmax computes softmax in-place over `x`.
// It subtracts the maximum value first to prevent numerical overflow (+Inf/NaN).
func Softmax(x []float32) error {
	if len(x) == 0 {
		return nil
	}

	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sum float64
	for i, v := range x {
		expVal := math.Exp(float64(v - maxVal))
		x[i] = float32(expVal)
		sum += expVal
	}

	if sum > 0 {
		invSum := float32(1.0 / sum)
		for i := range x {
			x[i] *= invSum
		}
	}
	return nil
}
