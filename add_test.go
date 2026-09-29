package nn

import (
	"testing"
)

func TestAdd_InPlace(t *testing.T) {
	dst := []float32{1.0, 2.0, 3.0}
	src := []float32{4.0, 5.0, 6.0}

	err := Add(dst, src)
	if err != nil {
		t.Fatalf("Add returned error: %v", err)
	}

	expected := []float32{5.0, 7.0, 9.0}
	for i, v := range dst {
		if v != expected[i] {
			t.Errorf("dst[%d] = %v, expected %v", i, v, expected[i])
		}
	}
}
