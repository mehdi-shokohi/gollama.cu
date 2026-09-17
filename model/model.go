// Package model runs a llama-architecture transformer from a GGUF file on
// the GPU: it uploads the weights, keeps a KV cache, and computes the
// logits of one token at a time (decode; batched prefill comes later).
package model

import (
	"context"
	"fmt"
	"math"

	"github.com/eitamring/gocudrv/cuda"

	"github.com/mehdi-shokohi/gollama/backend/gpu"
	"github.com/mehdi-shokohi/gollama/f16"
	"github.com/mehdi-shokohi/gollama/gguf"
	"github.com/mehdi-shokohi/gollama/quant"
)

// Config is the llama hyper-parameter set read from the GGUF metadata.
type Config struct {
	Arch     string
	Layers   int
	Dim      int // embedding length
	FFN      int // feed-forward length
	Heads    int
	KVHeads  int
	HeadDim  int
	Vocab    int
	CtxLen   int // the model's trained context length
	RopeBase float32
	Eps      float32
}

// ConfigFrom reads the config of a llama-architecture file.
func ConfigFrom(f *gguf.File) (Config, error) {
	arch := f.String("general.architecture", "")
	if arch != "llama" {
		return Config{}, fmt.Errorf("model: architecture %q is not llama", arch)
	}
	k := func(s string) string { return arch + "." + s }
	c := Config{
		Arch:     arch,
		Layers:   int(f.Uint(k("block_count"), 0)),
		Dim:      int(f.Uint(k("embedding_length"), 0)),
		FFN:      int(f.Uint(k("feed_forward_length"), 0)),
		Heads:    int(f.Uint(k("attention.head_count"), 0)),
		CtxLen:   int(f.Uint(k("context_length"), 2048)),
		RopeBase: float32(f.Float(k("rope.freq_base"), 10000)),
		Eps:      float32(f.Float(k("attention.layer_norm_rms_epsilon"), 1e-5)),
	}
	c.KVHeads = int(f.Uint(k("attention.head_count_kv"), uint64(c.Heads)))
	if c.Layers == 0 || c.Dim == 0 || c.FFN == 0 || c.Heads == 0 {
		return Config{}, fmt.Errorf("model: incomplete %s metadata", arch)
	}
	c.HeadDim = int(f.Uint(k("attention.key_length"), uint64(c.Dim/c.Heads)))
	if c.Heads%c.KVHeads != 0 {
		return Config{}, fmt.Errorf("model: %d heads is not a multiple of %d KV heads", c.Heads, c.KVHeads)
	}
	embd := f.Tensor("token_embd.weight")
	if embd == nil {
		return Config{}, fmt.Errorf("model: no token_embd.weight")
	}
	c.Vocab = embd.Shape()[0]
	return c, nil
}

// Linear is a [Rows][Cols] weight matrix on the device in its file
// format: quantized bytes, F16, or F32.
type Linear struct {
	Type       gguf.Type
	Rows, Cols int
	q          *cuda.Buffer[uint8]
	h          *cuda.Buffer[f16.Half]
	f          *cuda.Buffer[float32]
}

func uploadLinear(ctx *cuda.Context, t *gguf.TensorInfo) (*Linear, error) {
	shape := t.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("model: %s: not a matrix", t.Name)
	}
	l := &Linear{Type: t.Type, Rows: shape[0], Cols: shape[1]}
	data := t.Data()
	var err error
	switch t.Type {
	case gguf.F32:
		w := make([]float32, t.NumElems())
		if err = quant.Dequantize(gguf.F32, data, w); err == nil {
			l.f, err = alloc(ctx, w)
		}
	case gguf.F16:
		w := make([]f16.Half, t.NumElems())
		for i := range w {
			w[i] = f16.Half(uint16(data[2*i]) | uint16(data[2*i+1])<<8)
		}
		l.h, err = alloc(ctx, w)
	case gguf.Q4_0, gguf.Q4_K, gguf.Q6_K:
		l.q, err = alloc(ctx, data)
	default:
		return nil, fmt.Errorf("model: %s: no kernel for %s weights", t.Name, t.Type)
	}
	if err != nil {
		return nil, fmt.Errorf("model: %s: %w", t.Name, err)
	}
	return l, nil
}

// MatVec computes y[b] = W x[b] for batch vectors x.
func (l *Linear) MatVec(ctx context.Context, k *gpu.Kernels, s *cuda.Stream, x, y *cuda.Buffer[float32], batch int) error {
	switch {
	case l.q != nil:
		return k.MatVecQ(ctx, s, l.Type, l.q, x, y, l.Rows, l.Cols, batch)
	case l.h != nil:
		return k.MatVecF16(ctx, s, l.h, x, y, l.Rows, l.Cols, batch)
	default:
		return k.MatVecF32(ctx, s, l.f, x, y, l.Rows, l.Cols, batch)
	}
}

func (l *Linear) close() {
	for _, b := range []interface{ Close() error }{l.q, l.h, l.f} {
		if b != nil {
			b.Close()
		}
	}
}

type layer struct {
	attnNorm, ffnNorm              *cuda.Buffer[float32]
	wq, wk, wv, wo, gate, up, down *Linear
	kcache, vcache                 *cuda.Buffer[float32] // [maxLen][KVHeads*HeadDim]
}

// Model is a loaded model with a KV cache of MaxLen positions.
type Model struct {
	Config
	MaxLen int

	file   *gguf.File
	embd   *gguf.TensorInfo // token_embd.weight, read on the host per token
	k      *gpu.Kernels
	ctx    *cuda.Context
	layers []*layer
	norm   *cuda.Buffer[float32]
	output *Linear // LM head (token_embd when tied)
	freq   *cuda.Buffer[float32]

	// per-token work buffers; kv holds the new K, then the new V, before
	// each is copied into its cache row
	x, xn, q, kv, attn, o, gate, up, act, logits *cuda.Buffer[float32]
	pos                                          *cuda.Buffer[int32]
	hostRow                                      []float32
	hostLogits                                   []float32
	closers                                      []interface{ Close() error }
}

// Load uploads the weights of f into the context and allocates a KV cache
// of maxLen positions. f must stay open while the model is in use (token
// embeddings are read from it).
func Load(f *gguf.File, ctx *cuda.Context, k *gpu.Kernels, maxLen int) (*Model, error) {
	cfg, err := ConfigFrom(f)
	if err != nil {
		return nil, err
	}
	if maxLen <= 0 || maxLen > cfg.CtxLen {
		maxLen = cfg.CtxLen
	}
	m := &Model{Config: cfg, MaxLen: maxLen, file: f, embd: f.Tensor("token_embd.weight"), k: k, ctx: ctx}
	if err := m.load(); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

func (m *Model) load() error {
	c, ctx := m.Config, m.ctx
	kvDim := c.KVHeads * c.HeadDim
	tensor := func(name string) (*gguf.TensorInfo, error) {
		t := m.file.Tensor(name)
		if t == nil {
			return nil, fmt.Errorf("model: no tensor %s", name)
		}
		return t, nil
	}
	norm := func(name string, n int) (*cuda.Buffer[float32], error) {
		t, err := tensor(name)
		if err != nil {
			return nil, err
		}
		if t.NumElems() != n {
			return nil, fmt.Errorf("model: %s has %d elements, want %d", name, t.NumElems(), n)
		}
		w := make([]float32, n)
		if err := quant.Dequantize(t.Type, t.Data(), w); err != nil {
			return nil, fmt.Errorf("model: %s: %w", name, err)
		}
		b, err := alloc(ctx, w)
		if err != nil {
			return nil, err
		}
		m.closers = append(m.closers, b)
		return b, nil
	}
	linear := func(name string, rows, cols int) (*Linear, error) {
		t, err := tensor(name)
		if err != nil {
			return nil, err
		}
		l, err := uploadLinear(ctx, t)
		if err != nil {
			return nil, err
		}
		if l.Rows != rows || l.Cols != cols {
			l.close()
			return nil, fmt.Errorf("model: %s is [%d][%d], want [%d][%d]", name, l.Rows, l.Cols, rows, cols)
		}
		m.closers = append(m.closers, closerFunc(l.close))
		return l, nil
	}
	work := func(n int) (*cuda.Buffer[float32], error) {
		b, err := cuda.Alloc[float32](ctx, n)
		if err != nil {
			return nil, err
		}
		m.closers = append(m.closers, b)
		return b, nil
	}

	var err error
	if !m.embd.Type.Known() || m.embd.Shape()[1] != c.Dim {
		return fmt.Errorf("model: token_embd.weight %v is not [vocab][%d]", m.embd.Shape(), c.Dim)
	}
	m.hostRow = make([]float32, c.Dim)
	m.hostLogits = make([]float32, c.Vocab)

	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		l := &layer{}
		steps := []func() error{
			func() (err error) { l.attnNorm, err = norm(p+"attn_norm.weight", c.Dim); return },
			func() (err error) { l.ffnNorm, err = norm(p+"ffn_norm.weight", c.Dim); return },
			func() (err error) { l.wq, err = linear(p+"attn_q.weight", c.Heads*c.HeadDim, c.Dim); return },
			func() (err error) { l.wk, err = linear(p+"attn_k.weight", kvDim, c.Dim); return },
			func() (err error) { l.wv, err = linear(p+"attn_v.weight", kvDim, c.Dim); return },
			func() (err error) { l.wo, err = linear(p+"attn_output.weight", c.Dim, c.Heads*c.HeadDim); return },
			func() (err error) { l.gate, err = linear(p+"ffn_gate.weight", c.FFN, c.Dim); return },
			func() (err error) { l.up, err = linear(p+"ffn_up.weight", c.FFN, c.Dim); return },
			func() (err error) { l.down, err = linear(p+"ffn_down.weight", c.Dim, c.FFN); return },
			func() (err error) { l.kcache, err = work(m.MaxLen * kvDim); return },
			func() (err error) { l.vcache, err = work(m.MaxLen * kvDim); return },
		}
		for _, step := range steps {
			if err := step(); err != nil {
				return err
			}
		}
		m.layers = append(m.layers, l)
	}
	if m.norm, err = norm("output_norm.weight", c.Dim); err != nil {
		return err
	}
	head := m.file.Tensor("output.weight")
	if head == nil { // tied to the embeddings
		head = m.embd
	}
	if m.output, err = uploadLinear(ctx, head); err != nil {
		return err
	}
	m.closers = append(m.closers, closerFunc(m.output.close))
	if m.output.Rows != c.Vocab || m.output.Cols != c.Dim {
		return fmt.Errorf("model: LM head is [%d][%d], want [%d][%d]", m.output.Rows, m.output.Cols, c.Vocab, c.Dim)
	}

	// Llama 3.1+ rope frequency factors; ones when the file has none.
	if t := m.file.Tensor("rope_freqs.weight"); t != nil {
		if t.NumElems() != c.HeadDim/2 {
			return fmt.Errorf("model: rope_freqs.weight has %d elements, want %d", t.NumElems(), c.HeadDim/2)
		}
		if m.freq, err = norm("rope_freqs.weight", c.HeadDim/2); err != nil {
			return err
		}
	} else {
		if m.freq, err = gpu.Ones(ctx, c.HeadDim/2); err != nil {
			return err
		}
		m.closers = append(m.closers, m.freq)
	}

	bufs := []struct {
		dst **cuda.Buffer[float32]
		n   int
	}{
		{&m.x, c.Dim}, {&m.xn, c.Dim}, {&m.q, c.Heads * c.HeadDim}, {&m.kv, kvDim},
		{&m.attn, c.Heads * c.HeadDim}, {&m.o, c.Dim}, {&m.gate, c.FFN}, {&m.up, c.FFN}, {&m.act, c.FFN},
		{&m.logits, c.Vocab},
	}
	for _, b := range bufs {
		if *b.dst, err = work(b.n); err != nil {
			return err
		}
	}
	if m.pos, err = cuda.Alloc[int32](ctx, 1); err != nil {
		return err
	}
	m.closers = append(m.closers, m.pos)
	return nil
}

// Forward runs token at position pos (0-based; positions must be fed in
// order, each once) and returns its logits over the vocabulary. The
// returned slice is reused by the next call.
func (m *Model) Forward(ctx context.Context, token int32, pos int) ([]float32, error) {
	c := m.Config
	if pos < 0 || pos >= m.MaxLen {
		return nil, fmt.Errorf("model: position %d outside the %d-token cache", pos, m.MaxLen)
	}
	if token < 0 || int(token) >= c.Vocab {
		return nil, fmt.Errorf("model: token %d outside the vocabulary", token)
	}
	kvDim := c.KVHeads * c.HeadDim
	scale := float32(1 / math.Sqrt(float64(c.HeadDim)))
	k := m.k
	var s *cuda.Stream // the default stream keeps the host copies ordered

	if err := quant.Row(m.embd.Type, m.embd.Data(), c.Dim, int(token), m.hostRow); err != nil {
		return nil, err
	}
	if err := m.x.CopyFrom(ctx, m.hostRow); err != nil {
		return nil, err
	}
	if err := m.pos.CopyFrom(ctx, []int32{int32(pos)}); err != nil {
		return nil, err
	}
	for _, l := range m.layers {
		steps := []func() error{
			func() error { return k.RMSNorm(ctx, s, m.x, l.attnNorm, m.xn, 1, c.Dim, c.Eps) },
			func() error { return l.wq.MatVec(ctx, k, s, m.xn, m.q, 1) },
			func() error { return l.wk.MatVec(ctx, k, s, m.xn, m.kv, 1) },
			func() error { return k.RoPE(ctx, s, m.q, m.pos, m.freq, 1, c.Heads, c.HeadDim, c.RopeBase) },
			func() error { return k.RoPE(ctx, s, m.kv, m.pos, m.freq, 1, c.KVHeads, c.HeadDim, c.RopeBase) },
			func() error { return m.kv.CopyToDeviceAt(ctx, pos*kvDim, l.kcache, 0, kvDim) },
			func() error { return l.wv.MatVec(ctx, k, s, m.xn, m.kv, 1) },
			func() error { return m.kv.CopyToDeviceAt(ctx, pos*kvDim, l.vcache, 0, kvDim) },
			func() error {
				return k.AttnDecode(ctx, s, m.q, l.kcache, l.vcache, m.attn, pos+1, c.Heads, c.KVHeads, c.HeadDim, scale)
			},
			func() error { return l.wo.MatVec(ctx, k, s, m.attn, m.o, 1) },
			func() error { return k.Add(ctx, s, m.x, m.o, m.x, c.Dim) },
			func() error { return k.RMSNorm(ctx, s, m.x, l.ffnNorm, m.xn, 1, c.Dim, c.Eps) },
			func() error { return l.gate.MatVec(ctx, k, s, m.xn, m.gate, 1) },
			func() error { return l.up.MatVec(ctx, k, s, m.xn, m.up, 1) },
			func() error { return k.SiluMul(ctx, s, m.gate, m.up, m.act, c.FFN) },
			func() error { return l.down.MatVec(ctx, k, s, m.act, m.o, 1) },
			func() error { return k.Add(ctx, s, m.x, m.o, m.x, c.Dim) },
		}
		for _, step := range steps {
			if err := step(); err != nil {
				return nil, err
			}
		}
	}
	if err := k.RMSNorm(ctx, s, m.x, m.norm, m.xn, 1, c.Dim, c.Eps); err != nil {
		return nil, err
	}
	if err := m.output.MatVec(ctx, k, s, m.xn, m.logits, 1); err != nil {
		return nil, err
	}
	if err := m.logits.CopyTo(ctx, m.hostLogits); err != nil {
		return nil, err
	}
	return m.hostLogits, nil
}

// Argmax is greedy sampling.
func Argmax(logits []float32) int32 {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return int32(best)
}

// Close frees the device memory.
func (m *Model) Close() {
	for i := len(m.closers) - 1; i >= 0; i-- {
		m.closers[i].Close()
	}
	m.closers = nil
}

type closerFunc func()

func (f closerFunc) Close() error { f(); return nil }

func alloc[T cuda.Supported](ctx *cuda.Context, src []T) (*cuda.Buffer[T], error) {
	b, err := cuda.Alloc[T](ctx, len(src))
	if err != nil {
		return nil, err
	}
	if err := b.CopyFrom(context.Background(), src); err != nil {
		b.Close()
		return nil, err
	}
	return b, nil
}
