package quant

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/mehdi-shokohi/gollama.cu/f16"
	"github.com/mehdi-shokohi/gollama.cu/gguf"
)

// Llama32_3B is the Llama 3.2 3B Instruct blob of the local ollama store;
// tests that need a real model skip when it is absent.
const Llama32_3B = "/media/mate/ext/ollama/.ollama/models/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"

func TestQ8Q4(t *testing.T) {
	// one Q8_0 block and one Q4_0 block with d = 0.5
	q8 := make([]byte, 34)
	binary.LittleEndian.PutUint16(q8, uint16(f16.From(0.5)))
	for j := 0; j < 32; j++ {
		q8[2+j] = byte(int8(j - 16))
	}
	out := make([]float32, 32)
	if err := Dequantize(gguf.Q8_0, q8, out); err != nil {
		t.Fatal(err)
	}
	for j := range out {
		if out[j] != 0.5*float32(j-16) {
			t.Fatalf("q8[%d] = %v", j, out[j])
		}
	}
	q4 := make([]byte, 18)
	binary.LittleEndian.PutUint16(q4, uint16(f16.From(0.5)))
	for j := 0; j < 16; j++ {
		q4[2+j] = byte(j) | byte(15-j)<<4
	}
	if err := Dequantize(gguf.Q4_0, q4, out); err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 16; j++ {
		if out[j] != 0.5*float32(j-8) || out[j+16] != 0.5*float32(15-j-8) {
			t.Fatalf("q4[%d] = %v %v", j, out[j], out[j+16])
		}
	}
}

// TestRealModel decodes rows of every quant type in the local model and
// checks they are finite with a plausible magnitude.
func TestRealModel(t *testing.T) {
	if _, err := os.Stat(Llama32_3B); err != nil {
		t.Skip("no local model")
	}
	f, err := gguf.Open(Llama32_3B)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, name := range []string{"token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight", "blk.0.attn_norm.weight"} {
		ti := f.Tensor(name)
		cols := ti.Shape()[len(ti.Shape())-1]
		row := make([]float32, cols)
		if err := Row(ti.Type, ti.Data(), cols, 0, row); err != nil {
			t.Fatal(err)
		}
		var ss float64
		for _, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s: non-finite value", name)
			}
			ss += float64(v) * float64(v)
		}
		rms := math.Sqrt(ss / float64(cols))
		t.Logf("%s %s row 0: rms %.4g, first %v", name, ti.Type, rms, row[:4])
		if rms == 0 || rms > 10 {
			t.Errorf("%s: implausible rms %v", name, rms)
		}
	}
}
