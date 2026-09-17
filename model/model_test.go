package model

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/eitamring/gocudrv/cuda"

	"github.com/mehdi-shokohi/gollama/backend/cpu"
	"github.com/mehdi-shokohi/gollama/backend/gpu"
	"github.com/mehdi-shokohi/gollama/gguf"
	"github.com/mehdi-shokohi/gollama/ollama"
	"github.com/mehdi-shokohi/gollama/quant"
)

func setup(t *testing.T, name string) (*gguf.File, *cuda.Context, *gpu.Kernels) {
	t.Helper()
	path, err := ollama.Resolve(name)
	if err != nil {
		t.Skipf("model not on disk: %v", err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
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
	k, err := gpu.Load(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { k.Close() })
	return f, ctx, k
}

// TestLinearVsCPU multiplies real weight matrices of each local model on
// the GPU and on the host.
func TestLinearVsCPU(t *testing.T) {
	for _, name := range []string{"llama3.2", "llama3.1"} {
		t.Run(name, func(t *testing.T) {
			f, ctx, k := setup(t, name)
			rng := rand.New(rand.NewSource(1))
			for _, tn := range []string{"blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.3.ffn_down.weight", "blk.3.ffn_up.weight", "output.weight", "token_embd.weight"} {
				ti := f.Tensor(tn)
				if ti == nil {
					continue
				}
				l, err := uploadLinear(ctx, ti)
				if err != nil {
					t.Fatal(err)
				}
				defer l.close()
				x := make([]float32, l.Cols)
				for i := range x {
					x[i] = rng.Float32()*2 - 1
				}
				rows := l.Rows
				if rows > 20000 { // the vocab-sized matrices: a prefix is enough
					rows = 2048
				}
				want := make([]float32, l.Rows)
				if err := cpu.MatVecQ(ti.Type, ti.Data(), x, want, rows, l.Cols, 1); err != nil {
					t.Fatal(err)
				}
				dx, err := alloc(ctx, x)
				if err != nil {
					t.Fatal(err)
				}
				defer dx.Close()
				dy, err := cuda.Alloc[float32](ctx, l.Rows)
				if err != nil {
					t.Fatal(err)
				}
				defer dy.Close()
				if err := l.MatVec(context.Background(), k, nil, dx, dy, 1); err != nil {
					t.Fatal(err)
				}
				got := make([]float32, l.Rows)
				if err := dy.CopyTo(context.Background(), got); err != nil {
					t.Fatal(err)
				}
				worst := 0.0
				for r := 0; r < rows; r++ {
					d := math.Abs(float64(got[r] - want[r]))
					worst = max(worst, d)
					if d > 1e-3+1e-3*math.Abs(float64(want[r])) {
						t.Fatalf("%s %s row %d: gpu %v, cpu %v", tn, ti.Type, r, got[r], want[r])
					}
				}
				t.Logf("%s %s [%d][%d]: %d rows checked, max abs diff %.3g", tn, ti.Type, l.Rows, l.Cols, rows, worst)
			}
		})
	}
}

// cpuForward is a pure-Go forward pass over the same GGUF tensors, built
// from the backend/cpu twins: the reference for Model.Forward. kc/vc are
// the caller's KV caches, [layer][pos*kvDim].
func cpuForward(t *testing.T, f *gguf.File, c Config, token int32, pos int, kc, vc [][]float32) []float32 {
	t.Helper()
	kvDim := c.KVHeads * c.HeadDim
	tensor := func(name string) *gguf.TensorInfo {
		ti := f.Tensor(name)
		if ti == nil {
			t.Fatalf("no tensor %s", name)
		}
		return ti
	}
	vec := func(name string) []float32 {
		ti := tensor(name)
		out := make([]float32, ti.NumElems())
		if err := quant.Dequantize(ti.Type, ti.Data(), out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	matvec := func(name string, x []float32) []float32 {
		ti := tensor(name)
		rows, cols := ti.Shape()[0], ti.Shape()[1]
		y := make([]float32, rows)
		if err := cpu.MatVecQ(ti.Type, ti.Data(), x, y, rows, cols, 1); err != nil {
			t.Fatal(err)
		}
		return y
	}
	var freq []float32
	if f.Tensor("rope_freqs.weight") != nil {
		freq = vec("rope_freqs.weight")
	}
	x := make([]float32, c.Dim)
	embd := tensor("token_embd.weight")
	if err := quant.Row(embd.Type, embd.Data(), c.Dim, int(token), x); err != nil {
		t.Fatal(err)
	}
	xn := make([]float32, c.Dim)
	posv := []int32{int32(pos)}
	scale := float32(1 / math.Sqrt(float64(c.HeadDim)))
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		cpu.RMSNorm(x, vec(p+"attn_norm.weight"), xn, 1, c.Dim, c.Eps)
		q, k, v := matvec(p+"attn_q.weight", xn), matvec(p+"attn_k.weight", xn), matvec(p+"attn_v.weight", xn)
		cpu.RoPE(q, posv, freq, 1, c.Heads, c.HeadDim, c.RopeBase)
		cpu.RoPE(k, posv, freq, 1, c.KVHeads, c.HeadDim, c.RopeBase)
		copy(kc[i][pos*kvDim:], k)
		copy(vc[i][pos*kvDim:], v)
		attn := make([]float32, c.Heads*c.HeadDim)
		cpu.AttnDecode(q, kc[i], vc[i], attn, pos+1, c.Heads, c.KVHeads, c.HeadDim, scale)
		cpu.Add(x, matvec(p+"attn_output.weight", attn), x)
		cpu.RMSNorm(x, vec(p+"ffn_norm.weight"), xn, 1, c.Dim, c.Eps)
		gate, up := matvec(p+"ffn_gate.weight", xn), matvec(p+"ffn_up.weight", xn)
		cpu.SiluMul(gate, up, gate)
		cpu.Add(x, matvec(p+"ffn_down.weight", gate), x)
	}
	cpu.RMSNorm(x, vec("output_norm.weight"), xn, 1, c.Dim, c.Eps)
	head := "output.weight"
	if f.Tensor(head) == nil {
		head = "token_embd.weight"
	}
	return matvec(head, xn)
}

// TestForwardVsCPU compares the GPU logits of the first few prompt tokens
// with the pure-Go forward pass. Slow (a CPU forward per token); set
// GOLLAMA_LONG to run it on every local model rather than the 3B only.
func TestForwardVsCPU(t *testing.T) {
	names := []string{"llama3.2"}
	if os.Getenv("GOLLAMA_LONG") != "" {
		names = append(names, "llama3.1")
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			f, ctx, k := setup(t, name)
			m, err := Load(f, ctx, k, 16)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			kvDim := m.KVHeads * m.HeadDim
			kc, vc := make([][]float32, m.Layers), make([][]float32, m.Layers)
			for i := range kc {
				kc[i], vc[i] = make([]float32, 3*kvDim), make([]float32, 3*kvDim)
			}
			for pos, tok := range []int32{128000, 10445, 374} { // <|begin_of_text|> Why is
				got, err := m.Forward(context.Background(), tok, pos)
				if err != nil {
					t.Fatal(err)
				}
				got = append([]float32(nil), got...)
				want := cpuForward(t, f, m.Config, tok, pos, kc, vc)
				worst, wi := 0.0, 0
				for i := range got {
					if d := math.Abs(float64(got[i] - want[i])); d > worst {
						worst, wi = d, i
					}
				}
				t.Logf("pos %d token %d: argmax gpu %d cpu %d, max abs diff %.3g at %d (gpu %v cpu %v)",
					pos, tok, Argmax(got), Argmax(want), worst, wi, got[wi], want[wi])
				if Argmax(got) != Argmax(want) || worst > 0.05 {
					t.Errorf("pos %d: logits differ", pos)
				}
			}
		})
	}
}
