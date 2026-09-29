// Package nn holds the stateless operations every webtyp model computes: matrix
// multiplication, normalization, activations, softmax and rotary position embeddings. They work
// on plain []float32 slices with explicit shapes, in plain Go, for TinyGo/WASM.
package nn

import (
	"webtyp.com/fmt"
	"webtyp.com/vector"
)

// MatmulT computes C[m,n] = A[m,k] · Bᵀ[n,k], i.e. B is stored TRANSPOSED so every
// inner product is over contiguous memory — which is what vector.Dot is fast at.
func MatmulT(dst, a, bT []float32, m, k, n int) error {
	if m <= 0 || k <= 0 || n <= 0 {
		return fmt.Err("nn: invalid matmul dimensions")
	}
	if len(a) < m*k || len(bT) < n*k || len(dst) < m*n {
		return fmt.Err("nn: buffer too short for matmul")
	}

	for i := 0; i < m; i++ {
		rowA := a[i*k : (i+1)*k]
		outRow := dst[i*n : (i+1)*n]
		for j := 0; j < n; j++ {
			rowB := bT[j*k : (j+1)*k]
			outRow[j] = vector.Dot(rowA, rowB)
		}
	}
	return nil
}
