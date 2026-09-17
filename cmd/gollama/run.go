package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/eitamring/gocudrv/cuda"

	"github.com/mehdi-shokohi/gollama.cu/backend/gpu"
	"github.com/mehdi-shokohi/gollama.cu/gguf"
	"github.com/mehdi-shokohi/gollama.cu/model"
	"github.com/mehdi-shokohi/gollama.cu/ollama"
	"github.com/mehdi-shokohi/gollama.cu/tokenizer"
)

func run(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	modelRef := fs.String("m", "", "model: a GGUF file, an ollama manifest, or an ollama model name (llama3.1, llama3.2:3b)")
	prompt := fs.String("p", "", "the prompt")
	system := fs.String("system", "", "system prompt (chat mode)")
	raw := fs.Bool("raw", false, "no chat template: BOS + prompt")
	n := fs.Int("n", 128, "maximum tokens to generate")
	ctxLen := fs.Int("ctx", 2048, "KV cache length (0 = the model's context length)")
	verbose := fs.Bool("v", false, "print the model config and token ids to stderr")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: gollama run [-m] <model> -p <prompt> [-n tokens] [-system text] [-raw] [-ctx len] [-v]")
		fs.PrintDefaults()
	}
	var positional string
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") { // gollama run llama3.2 -p ...
		positional, args = args[0], args[1:]
	}
	fs.Parse(args)
	if *modelRef == "" {
		*modelRef = positional
	}
	if *modelRef == "" || *prompt == "" {
		fs.Usage()
		os.Exit(2)
	}
	ctx := context.Background()

	path, err := ollama.Resolve(*modelRef)
	if err != nil {
		return err
	}
	f, err := gguf.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	tk, err := tokenizer.FromGGUF(f)
	if err != nil {
		return err
	}

	if err := cuda.Init(); err != nil {
		return fmt.Errorf("CUDA: %w", err)
	}
	dev, err := cuda.GetDevice(0)
	if err != nil {
		return err
	}
	cctx, err := dev.Primary()
	if err != nil {
		return err
	}
	defer cctx.Close()
	k, err := gpu.Load(cctx)
	if err != nil {
		return err
	}
	defer k.Close()

	t0 := time.Now()
	m, err := model.Load(f, cctx, k, *ctxLen)
	if err != nil {
		return err
	}
	defer m.Close()
	if *verbose {
		fmt.Fprintf(os.Stderr, "%s: %s, %d layers, dim %d, ffn %d, %d/%d heads x %d, vocab %d, cache %d, loaded in %.1fs\n",
			path, f.String("general.name", "?"), m.Layers, m.Dim, m.FFN, m.Heads, m.KVHeads, m.HeadDim, m.Vocab, m.MaxLen,
			time.Since(t0).Seconds())
	}

	ids := promptTokens(tk, *prompt, *system, *raw)
	if *verbose {
		fmt.Fprintf(os.Stderr, "prompt: %d tokens %v\n", len(ids), ids)
	}
	if len(ids)+*n > m.MaxLen {
		return fmt.Errorf("prompt (%d) + generation (%d) exceeds the %d-token cache; raise -ctx", len(ids), *n, m.MaxLen)
	}
	stop := map[int32]bool{tk.EOS: true}
	if id := tk.ID("<|end_of_text|>"); id >= 0 {
		stop[id] = true
	}

	// Decode-only: the prompt is fed one token at a time.
	t0 = time.Now()
	var logits []float32
	for i, id := range ids {
		if logits, err = m.Forward(ctx, id, i); err != nil {
			return err
		}
	}
	prefill := time.Since(t0)

	t0 = time.Now()
	var out utf8Writer
	generated := 0
	for pos := len(ids); generated < *n; pos++ {
		next := model.Argmax(logits)
		if stop[next] {
			break
		}
		generated++
		out.Write(tk.Piece(next))
		if *verbose {
			fmt.Fprintf(os.Stderr, "[%d]", next)
		}
		if logits, err = m.Forward(ctx, next, pos); err != nil {
			return err
		}
	}
	out.Flush()
	fmt.Println()
	gen := time.Since(t0)
	fmt.Fprintf(os.Stderr, "\n%d prompt tokens in %.2fs (%.1f tok/s), %d generated in %.2fs (%.1f tok/s)\n",
		len(ids), prefill.Seconds(), float64(len(ids))/prefill.Seconds(),
		generated, gen.Seconds(), float64(generated)/gen.Seconds())
	return nil
}

// promptTokens builds the token sequence: BOS + prompt in raw mode, else
// the llama 3 chat format
//
//	<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
//
// with a system turn first when system is set.
func promptTokens(tk *tokenizer.Tokenizer, prompt, system string, raw bool) []int32 {
	ids := []int32{tk.BOS}
	if raw {
		return append(ids, tk.Encode(prompt)...)
	}
	sh, eh, eot := tk.ID("<|start_header_id|>"), tk.ID("<|end_header_id|>"), tk.ID("<|eot_id|>")
	turn := func(role, content string) {
		ids = append(ids, sh)
		ids = append(ids, tk.Encode(role)...)
		ids = append(ids, eh)
		ids = append(ids, tk.Encode("\n\n"+content)...) // tokenized together, as the template text is
	}
	if system != "" {
		turn("system", system)
		ids = append(ids, eot)
	}
	turn("user", prompt)
	ids = append(ids, eot)
	turn("assistant", "")
	return ids
}

// utf8Writer prints token pieces to stdout, holding back the bytes of an
// incomplete UTF-8 sequence until the next piece completes it.
type utf8Writer struct{ pending []byte }

func (w *utf8Writer) Write(p []byte) {
	w.pending = append(w.pending, p...)
	// keep any trailing partial rune
	keep := 0
	for i := len(w.pending) - 1; i >= 0 && i >= len(w.pending)-utf8.UTFMax; i-- {
		if utf8.RuneStart(w.pending[i]) {
			if !utf8.FullRune(w.pending[i:]) {
				keep = len(w.pending) - i
			}
			break
		}
	}
	os.Stdout.Write(w.pending[:len(w.pending)-keep])
	w.pending = append(w.pending[:0], w.pending[len(w.pending)-keep:]...)
}

func (w *utf8Writer) Flush() {
	if len(w.pending) > 0 {
		os.Stdout.Write([]byte(strings.ToValidUTF8(string(w.pending), "�")))
		w.pending = nil
	}
}
