package nn

import (
	"webtyp.com/fmt"
)

// Add performs element-wise in-place addition: dst[i] += src[i].
func Add(dst, src []float32) error {
	if len(dst) != len(src) {
		return fmt.Err("nn: length mismatch for add")
	}
	for i, v := range src {
		dst[i] += v
	}
	return nil
}
