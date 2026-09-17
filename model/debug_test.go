package model

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/eitamring/gocudrv/cuda"

	"github.com/mehdi-shokohi/gollama/backend/cpu"
	"github.com/mehdi-shokohi/gollama/quant"
)

func TestBisectLayer0(t *testing.T) {
	f, ctx, k := setup(t, "llama3.1")
	m, err := Load(f, ctx, k, 16)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	c := m.Config
	l := m.layers[0]
	bg := context.Background()
	dl := func(b *cuda.Buffer[float32]) []float32 {
		out := make([]float32, b.Len())
		if err := b.CopyTo(bg, out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	cmp := func(name string, got, want []float32) {
		worst := 0.0
		for i := range got {
			worst = max(worst, math.Abs(float64(got[i]-want[i])))
		}
		t.Logf("%-8s max abs diff %.3g  (gpu[:3] %v cpu[:3] %v)", name, worst, got[:3], want[:3])
	}
	vec := func(name string) []float32 {
		ti := f.Tensor(name)
		out := make([]float32, ti.NumElems())
		if err := quant.Dequantize(ti.Type, ti.Data(), out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	matvec := func(name string, x []float32) []float32 {
		ti := f.Tensor(name)
		rows, cols := ti.Shape()[0], ti.Shape()[1]
		y := make([]float32, rows)
		if err := cpu.MatVecQ(ti.Type, ti.Data(), x, y, rows, cols, 1); err != nil {
			t.Fatal(err)
		}
		return y
	}
	// host side
	x := make([]float32, c.Dim)
	embd := f.Tensor("token_embd.weight")
	if err := quant.Row(embd.Type, embd.Data(), c.Dim, 10445, x); err != nil {
		t.Fatal(err)
	}
	xn := make([]float32, c.Dim)
	cpu.RMSNorm(x, vec("blk.0.attn_norm.weight"), xn, 1, c.Dim, c.Eps)
	q := matvec("blk.0.attn_q.weight", xn)
	gate := matvec("blk.0.ffn_gate.weight", xn)
	up := matvec("blk.0.ffn_up.weight", xn)
	act := make([]float32, c.FFN)
	cpu.SiluMul(gate, up, act)
	down := matvec("blk.0.ffn_down.weight", act)

	// device side
	if err := m.x.CopyFrom(bg, x); err != nil {
		t.Fatal(err)
	}
	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	must(k.RMSNorm(bg, nil, m.x, l.attnNorm, m.xn, 1, c.Dim, c.Eps))
	cmp("rmsnorm", dl(m.xn), xn)
	must(l.wq.MatVec(bg, k, nil, m.xn, m.q, 1))
	cmp("q", dl(m.q), q)
	must(l.gate.MatVec(bg, k, nil, m.xn, m.gate, 1))
	cmp("gate", dl(m.gate), gate)
	must(l.up.MatVec(bg, k, nil, m.xn, m.up, 1))
	cmp("up", dl(m.up), up)
	must(k.SiluMul(bg, nil, m.gate, m.up, m.act, c.FFN))
	cmp("act", dl(m.act), act)
	must(l.down.MatVec(bg, k, nil, m.act, m.o, 1))
	cmp("down", dl(m.o), down)
	// attention at pos 0
	kvDim := c.KVHeads * c.HeadDim
	kk, vv := matvec("blk.0.attn_k.weight", xn), matvec("blk.0.attn_v.weight", xn)
	attn := make([]float32, c.Heads*c.HeadDim)
	cpu.AttnDecode(q, kk, vv, attn, 1, c.Heads, c.KVHeads, c.HeadDim, float32(1/math.Sqrt(float64(c.HeadDim))))
	o := matvec("blk.0.attn_output.weight", attn)
	must(m.pos.CopyFrom(bg, []int32{0}))
	must(l.wq.MatVec(bg, k, nil, m.xn, m.q, 1))
	must(l.wk.MatVec(bg, k, nil, m.xn, m.kv, 1))
	cmp("k", dl(m.kv), kk)
	must(k.RoPE(bg, nil, m.q, m.pos, m.freq, 1, c.Heads, c.HeadDim, c.RopeBase))
	cmp("q rope0", dl(m.q), q)
	must(k.RoPE(bg, nil, m.kv, m.pos, m.freq, 1, c.KVHeads, c.HeadDim, c.RopeBase))
	cmp("k rope0", dl(m.kv), kk)
	must(m.kv.CopyToDeviceAt(bg, 0, l.kcache, 0, kvDim))
	cmp("kcache", dl(l.kcache)[:kvDim], kk)
	must(l.wv.MatVec(bg, k, nil, m.xn, m.kv, 1))
	must(m.kv.CopyToDeviceAt(bg, 0, l.vcache, 0, kvDim))
	cmp("vcache", dl(l.vcache)[:kvDim], vv)
	must(k.AttnDecode(bg, nil, m.q, l.kcache, l.vcache, m.attn, 1, c.Heads, c.KVHeads, c.HeadDim, float32(1/math.Sqrt(float64(c.HeadDim)))))
	cmp("attn", dl(m.attn), attn)
	must(l.wo.MatVec(bg, k, nil, m.attn, m.o, 1))
	cmp("o", dl(m.o), o)
	cmp("freq", dl(m.freq), make([]float32, c.HeadDim/2))

	// same matvec with the CPU's xn uploaded, to separate input from kernel error
	must(m.xn.CopyFrom(bg, xn))
	must(l.wq.MatVec(bg, k, nil, m.xn, m.q, 1))
	cmp("q(cpu xn)", dl(m.q), q)
	fmt.Println()
}
