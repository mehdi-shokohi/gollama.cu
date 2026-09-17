package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	w := NewWriter()
	w.Set("general.architecture", "llama")
	w.Set("llama.block_count", uint32(2))
	w.Set("llama.rope.freq_base", float32(10000))
	w.Set("tokenizer.ggml.tokens", []string{"<s>", "</s>", "hello"})
	w.Set("tokenizer.ggml.scores", []float32{0, -1, 2.5})
	w.Set("x.flag", true)
	w.Set("x.big", int64(-7))

	a := make([]byte, 4*4*3) // f32 [4,3]
	for i := 0; i < 12; i++ {
		binary.LittleEndian.PutUint32(a[i*4:], math.Float32bits(float32(i)))
	}
	if err := w.AddTensor("a.weight", []int{4, 3}, F32, a); err != nil {
		t.Fatal(err)
	}
	q := make([]byte, Q8_0.RowBytes(64)*2) // q8_0 [2,64]
	for i := range q {
		q[i] = byte(i)
	}
	if err := w.AddTensor("q.weight", []int{2, 64}, Q8_0, q); err != nil {
		t.Fatal(err)
	}
	if err := w.AddTensor("bad", []int{2, 63}, Q8_0, q); err == nil {
		t.Fatal("expected block-size error")
	}

	path := filepath.Join(t.TempDir(), "m.gguf")
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if f.Version != 3 || f.Alignment != 32 {
		t.Errorf("version %d alignment %d", f.Version, f.Alignment)
	}
	if got := f.String("general.architecture", ""); got != "llama" {
		t.Errorf("arch = %q", got)
	}
	if got := f.Uint("llama.block_count", 0); got != 2 {
		t.Errorf("block_count = %d", got)
	}
	if got := f.Float("llama.rope.freq_base", 0); got != 10000 {
		t.Errorf("freq_base = %v", got)
	}
	if got := f.Strings("tokenizer.ggml.tokens"); len(got) != 3 || got[2] != "hello" {
		t.Errorf("tokens = %v", got)
	}
	v, _ := f.Get("tokenizer.ggml.scores")
	if got := v.Floats(); len(got) != 3 || got[2] != 2.5 {
		t.Errorf("scores = %v", got)
	}
	if v, _ := f.Get("x.flag"); v.V != true {
		t.Errorf("flag = %v", v)
	}
	if v, _ := f.Get("x.big"); int64(v.Uint64()) != -7 {
		t.Errorf("big = %v", v)
	}
	if len(f.Keys) != 7 || f.Keys[0] != "general.architecture" {
		t.Errorf("keys = %v", f.Keys)
	}

	ta := f.Tensor("a.weight")
	if ta == nil || ta.Type != F32 || ta.NumElems() != 12 || ta.Bytes() != 48 {
		t.Fatalf("a.weight = %v", ta)
	}
	if s := ta.Shape(); s[0] != 4 || s[1] != 3 || ta.Dims[0] != 3 {
		t.Errorf("shape %v dims %v", s, ta.Dims)
	}
	if !bytes.Equal(ta.Data(), a) {
		t.Error("a.weight data mismatch")
	}
	tq := f.Tensor("q.weight")
	if tq == nil || tq.Type != Q8_0 || tq.Bytes() != 136 || tq.Offset%32 != 0 {
		t.Fatalf("q.weight = %v offset %d", tq, tq.Offset)
	}
	if !bytes.Equal(tq.Data(), q) {
		t.Error("q.weight data mismatch")
	}
	if f.Tensor("nope") != nil {
		t.Error("unexpected tensor")
	}
}

func TestCorrupt(t *testing.T) {
	w := NewWriter()
	w.Set("k", "v")
	if err := w.AddTensor("t", []int{8}, F32, make([]byte, 32)); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := w.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	good := buf.Bytes()
	if _, err := Parse(good); err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{0, 3, 8, 20, 40, len(good) - 1} {
		if _, err := Parse(good[:n]); err == nil {
			t.Errorf("Parse(good[:%d]) succeeded", n)
		}
	}
	bad := append([]byte(nil), good...)
	bad[0] = 'X'
	if _, err := Parse(bad); err == nil {
		t.Error("bad magic accepted")
	}
}
