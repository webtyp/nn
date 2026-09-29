package nn

import (
	"math"
	"testing"
)

func TestSoftmax_SumsToOne(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	err := Softmax(x)
	if err != nil {
		t.Fatalf("Softmax returned unexpected error: %v", err)
	}

	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax sum = %v, expected ~1.0", sum)
	}
}

func TestSoftmax_LargeLogitsNoNaN(t *testing.T) {
	x := []float32{-100.0, 100.0, 50.0, -50.0, 1000.0}
	err := Softmax(x)
	if err != nil {
		t.Fatalf("Softmax returned unexpected error: %v", err)
	}

	for i, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Softmax produce NaN or Inf at index %d: %v", i, v)
		}
	}

	// The max element (1000.0) should dominate and sum should be 1.0
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax sum = %v, expected ~1.0", sum)
	}
}

func TestSoftmax_Uniform(t *testing.T) {
	n := 5
	x := []float32{2.5, 2.5, 2.5, 2.5, 2.5}
	err := Softmax(x)
	if err != nil {
		t.Fatalf("Softmax returned unexpected error: %v", err)
	}

	expected := float32(1.0 / float32(n))
	for i, v := range x {
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("Index %d: got %v, expected %v", i, v, expected)
		}
	}
}
