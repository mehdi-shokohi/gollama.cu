package f16

import (
	"math"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	// every half value except NaNs survives half -> float32 -> half exactly
	for i := 0; i < 1<<16; i++ {
		h := Half(i)
		if h&0x7c00 == 0x7c00 && h&0x3ff != 0 {
			continue
		}
		if got := From(h.Float32()); got != h {
			t.Fatalf("0x%04x -> %v -> 0x%04x", uint16(h), h.Float32(), uint16(got))
		}
	}
}

func TestKnown(t *testing.T) {
	cases := []struct {
		x float32
		h Half
	}{
		{0, 0x0000}, {1, 0x3c00}, {-2, 0xc000}, {65504, 0x7bff}, {1e-8, 0x0000},
		{5.960464e-8, 0x0001}, {0.1, 0x2e66}, {float32(math.Inf(1)), 0x7c00},
		{1e5, 0x7c00}, {3.14159, 0x4248},
	}
	for _, c := range cases {
		if got := From(c.x); got != c.h {
			t.Errorf("From(%v) = 0x%04x, want 0x%04x", c.x, uint16(got), uint16(c.h))
		}
	}
	if !math.IsNaN(float64(From(float32(math.NaN())).Float32())) {
		t.Error("NaN lost")
	}
}
