package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mehdi-shokohi/gollama/backend/cpu"
)

func TestAttnShapes(t *testing.T) {
	k, ctx := setup(t)
	rng := rand.New(rand.NewSource(7))
	for _, sh := range []struct{ nHeads, nKV, headDim, kvLen int; qs float32 }{
		{6, 2, 64, 1, 1}, {32, 8, 128, 1, 1}, {32, 8, 64, 1, 1}, {8, 8, 128, 1, 1}, {24, 8, 128, 1, 1}, {6, 2, 64, 1, 10}, {24, 8, 128, 1, 10}, {24, 8, 128, 5, 10},
	} {
		nHeads, nKV, headDim, kvLen := sh.nHeads, sh.nKV, sh.headDim, sh.kvLen
		q := randf(rng, nHeads*headDim, sh.qs)
		kc, vc := randf(rng, kvLen*nKV*headDim, sh.qs), randf(rng, kvLen*nKV*headDim, 1)
		scale := float32(1 / math.Sqrt(float64(headDim)))
		want := make([]float32, nHeads*headDim)
		cpu.AttnDecode(q, kc, vc, want, kvLen, nHeads, nKV, headDim, scale)
		dq, dk, dv, dout := upload(t, k, q), upload(t, k, kc), upload(t, k, vc), upload(t, k, make([]float32, nHeads*headDim))
		if err := k.AttnDecode(ctx, nil, dq, dk, dv, dout, kvLen, nHeads, nKV, headDim, scale); err != nil {
			t.Fatal(err)
		}
		got := download(t, k, dout)
		worst := 0.0
		var bad []int
		for h := 0; h < nHeads; h++ {
			hw := 0.0
			for i := h * headDim; i < (h+1)*headDim; i++ {
				hw = max(hw, math.Abs(float64(got[i]-want[i])))
			}
			if hw > 1e-4 {
				bad = append(bad, h)
			}
			worst = max(worst, hw)
		}
		t.Logf("%+v: max abs diff %.3g, bad heads %v", sh, worst, bad)
	}
}
