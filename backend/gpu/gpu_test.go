package gpu

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/eitamring/gocudrv/cuda"

	"github.com/mehdi-shokohi/gollama/backend/cpu"
	"github.com/mehdi-shokohi/gollama/f16"
	"github.com/mehdi-shokohi/gollama/gguf"
)

// Every kernel is checked against its backend/cpu twin on random data.

func setup(t *testing.T) (*Kernels, context.Context) {
	t.Helper()
	if err := cuda.Init(); err != nil {
		t.Skipf("no CUDA: %v", err)
	}
	dev, err := cuda.GetDevice(0)
	if err != nil {
		t.Skipf("no GPU: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ctx.Close() })
	k, err := Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { k.Close() })
	return k, context.Background()
}

func upload[T cuda.Supported](t *testing.T, k *Kernels, src []T) *cuda.Buffer[T] {
	t.Helper()
	d, err := cuda.Alloc[T](k.ctx, len(src))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Close() })
	if err := d.CopyFrom(context.Background(), src); err != nil {
		t.Fatal(err)
	}
	return d
}

func download[T cuda.Supported](t *testing.T, k *Kernels, d *cuda.Buffer[T]) []T {
	t.Helper()
	if err := k.ctx.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	out := make([]T, d.Len())
	if err := d.CopyTo(context.Background(), out); err != nil {
		t.Fatal(err)
	}
	return out
}

func randf(rng *rand.Rand, n int, scale float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = (rng.Float32()*2 - 1) * scale
	}
	return s
}

// assertClose fails on the first element where |got-want| > atol + rtol*|want|.
func assertClose(t *testing.T, name string, got, want []float32, atol, rtol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	worst := 0.0
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		d := math.Abs(g - w)
		if d > atol+rtol*math.Abs(w) || math.IsNaN(g) {
			t.Fatalf("%s[%d] = %v, want %v (diff %g)", name, i, g, w, d)
		}
		worst = max(worst, d)
	}
	t.Logf("%s: %d elements, max abs diff %.3g", name, len(got), worst)
}

func TestRMSNorm(t *testing.T) {
	k, ctx := setup(t)
	rng := rand.New(rand.NewSource(1))
	const rows, n = 7, 2048 // TinyLlama's hidden size, an odd number of rows
	x, w := randf(rng, rows*n, 3), randf(rng, n, 1)
	want := make([]float32, rows*n)
	cpu.RMSNorm(x, w, want, rows, n, 1e-5)
	dx, dw, dout := upload(t, k, x), upload(t, k, w), upload(t, k, make([]float32, rows*n))
	if err := k.RMSNorm(ctx, nil, dx, dw, dout, rows, n, 1e-5); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "RMSNorm", download(t, k, dout), want, 1e-5, 1e-4)
	// in place
	if err := k.RMSNorm(ctx, nil, dx, dw, dx, rows, n, 1e-5); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "RMSNorm(in place)", download(t, k, dx), want, 1e-5, 1e-4)
}

func TestSoftmax(t *testing.T) {
	k, ctx := setup(t)
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{1, 31, 256, 1000, 32000} { // 32000 = the llama vocab
		const rows = 3
		x := randf(rng, rows*n, 10)
		want := make([]float32, rows*n)
		cpu.Softmax(x, want, rows, n)
		dx := upload(t, k, x)
		if err := k.Softmax(ctx, nil, dx, dx, rows, n); err != nil {
			t.Fatal(err)
		}
		got := download(t, k, dx)
		assertClose(t, "Softmax", got, want, 1e-7, 1e-4)
		for r := 0; r < rows; r++ {
			var sum float64
			for _, v := range got[r*n : (r+1)*n] {
				sum += float64(v)
			}
			if math.Abs(sum-1) > 1e-4 {
				t.Errorf("n=%d row %d sums to %v", n, r, sum)
			}
		}
	}
}

func TestElementwise(t *testing.T) {
	k, ctx := setup(t)
	rng := rand.New(rand.NewSource(3))
	const n = 5632*4 + 13 // TinyLlama's intermediate size, not a multiple of the block
	a, b := randf(rng, n, 8), randf(rng, n, 8)
	da, db, dout := upload(t, k, a), upload(t, k, b), upload(t, k, make([]float32, n))

	want := make([]float32, n)
	cpu.SiluMul(a, b, want)
	if err := k.SiluMul(ctx, nil, da, db, dout, n); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "SiluMul", download(t, k, dout), want, 1e-5, 1e-5)

	cpu.Add(a, b, want)
	if err := k.Add(ctx, nil, da, db, dout, n); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "Add", download(t, k, dout), want, 0, 0)
}

func TestRoPE(t *testing.T) {
	k, ctx := setup(t)
	rng := rand.New(rand.NewSource(4))
	const rows, nHeads, headDim = 5, 4, 64
	x := randf(rng, rows*nHeads*headDim, 2)
	pos := []int32{0, 1, 2, 1000, 4095}
	freq := randf(rng, headDim/2, 1)
	for i := range freq {
		freq[i] = 1 + 4*float32(math.Abs(float64(freq[i]))) // Llama 3 factors are in [1, 32]
	}
	want := append([]float32(nil), x...)
	cpu.RoPE(want, pos, freq, rows, nHeads, headDim, 10000)
	dx, dpos, dfreq := upload(t, k, x), upload(t, k, pos), upload(t, k, freq)
	if err := k.RoPE(ctx, nil, dx, dpos, dfreq, rows, nHeads, headDim, 10000); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "RoPE", download(t, k, dx), want, 1e-3, 1e-3) // libdevice sincos at theta ~ 4095

	ones, err := Ones(k.ctx, headDim/2)
	if err != nil {
		t.Fatal(err)
	}
	defer ones.Close()
	want = append(want[:0], x...)
	cpu.RoPE(want, pos, nil, rows, nHeads, headDim, 10000)
	dx = upload(t, k, x)
	if err := k.RoPE(ctx, nil, dx, dpos, ones, rows, nHeads, headDim, 10000); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "RoPE(no factors)", download(t, k, dx), want, 1e-3, 1e-3)
}

func TestMatVec(t *testing.T) {
	k, ctx := setup(t)
	rng := rand.New(rand.NewSource(5))
	const rows, cols, batch = 300, 2048, 3 // rows not a multiple of the 8 warps per block
	w, x := randf(rng, rows*cols, 0.05), randf(rng, batch*cols, 1)
	hw := make([]f16.Half, len(w))
	f16.FromFloat32(hw, w)
	f16.ToFloat32(w, hw) // the reference sees the same rounded weights

	want := make([]float32, batch*rows)
	cpu.MatVecF16(hw, x, want, rows, cols, batch)
	dw, dx, dy := upload(t, k, hw), upload(t, k, x), upload(t, k, make([]float32, batch*rows))
	if err := k.MatVecF16(ctx, nil, dw, dx, dy, rows, cols, batch); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "MatVecF16", download(t, k, dy), want, 1e-4, 1e-4)

	cpu.MatVecF32(w, x, want, rows, cols, batch)
	dw32 := upload(t, k, w)
	if err := k.MatVecF32(ctx, nil, dw32, dx, dy, rows, cols, batch); err != nil {
		t.Fatal(err)
	}
	assertClose(t, "MatVecF32", download(t, k, dy), want, 1e-4, 1e-4)
}

// randQuant makes a random [rows][cols] weight in typ's block format:
// random quants, and block scales drawn small enough to keep the dot
// products O(1).
func randQuant(rng *rand.Rand, typ gguf.Type, rows, cols int) []byte {
	w := make([]byte, typ.RowBytes(cols)*rows)
	rng.Read(w)
	bs := typ.TypeSize()
	scale := func(off int, mag float32) {
		for b := off; b < len(w); b += bs {
			h := f16.From((rng.Float32()*2 - 1) * mag)
			w[b], w[b+1] = byte(h), byte(h>>8)
		}
	}
	switch typ {
	case gguf.Q4_0:
		scale(0, 0.02)
	case gguf.Q4_K:
		scale(0, 0.002) // d: times a 6-bit sub-scale and a 4-bit quant
		scale(2, 0.002) // dmin
	case gguf.Q6_K:
		scale(208, 0.001) // d: times an int8 scale and a 6-bit quant
	}
	return w
}

func TestMatVecQuant(t *testing.T) {
	k, ctx := setup(t)
	rng := rand.New(rand.NewSource(6))
	const rows, batch = 300, 2
	for _, cols := range []int{1024, 4096} { // one block per lane, then several
		x := randf(rng, batch*cols, 1)
		dx, dy := upload(t, k, x), upload(t, k, make([]float32, batch*rows))
		for _, typ := range []gguf.Type{gguf.Q4_0, gguf.Q4_K, gguf.Q6_K} {
			w := randQuant(rng, typ, rows, cols)
			want := make([]float32, batch*rows)
			if err := cpu.MatVecQ(typ, w, x, want, rows, cols, batch); err != nil {
				t.Fatal(err)
			}
			dw := upload(t, k, w)
			if err := k.MatVecQ(ctx, nil, typ, dw, dx, dy, rows, cols, batch); err != nil {
				t.Fatal(err)
			}
			assertClose(t, "MatVec"+typ.String(), download(t, k, dy), want, 1e-4, 1e-4)
		}
	}
}

func TestAttnDecode(t *testing.T) {
	k, ctx := setup(t)
	rng := rand.New(rand.NewSource(7))
	const nHeads, nKV, headDim = 6, 2, 64 // GQA group 3
	for _, kvLen := range []int{1, 7, 256, 1000} {
		q := randf(rng, nHeads*headDim, 1)
		kc, vc := randf(rng, kvLen*nKV*headDim, 1), randf(rng, kvLen*nKV*headDim, 1)
		scale := float32(1 / math.Sqrt(headDim))
		want := make([]float32, nHeads*headDim)
		cpu.AttnDecode(q, kc, vc, want, kvLen, nHeads, nKV, headDim, scale)
		dq, dk, dv, dout := upload(t, k, q), upload(t, k, kc), upload(t, k, vc), upload(t, k, make([]float32, nHeads*headDim))
		if err := k.AttnDecode(ctx, nil, dq, dk, dv, dout, kvLen, nHeads, nKV, headDim, scale); err != nil {
			t.Fatal(err)
		}
		assertClose(t, "AttnDecode", download(t, k, dout), want, 1e-5, 1e-4)
	}
}
