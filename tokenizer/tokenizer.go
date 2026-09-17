// Package tokenizer is the byte-level BPE tokenizer of the llama 3 family
// (tokenizer.ggml.model = gpt2, pre = llama-bpe), built from the vocabulary
// and merge list in a GGUF file.
//
// Text is first split by the llama-bpe pre-tokenizer (tiktoken's cl100k
// regex, hand-coded here because Go's regexp has no lookahead), each
// piece's bytes are mapped to the GPT-2 printable alphabet, merged by
// rank, and the merged pieces are looked up in the vocabulary.
package tokenizer

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/mehdi-shokohi/gollama/gguf"
)

// Token types of tokenizer.ggml.token_type (llama.cpp's llama_token_attr).
const (
	Normal      = 1
	Unknown     = 2
	Control     = 3
	UserDefined = 4
	Unused      = 5
	Byte        = 6
)

// Tokenizer encodes text to token ids and back.
type Tokenizer struct {
	tokens []string // id -> token in the GPT-2 byte alphabet
	types  []int32
	ids    map[string]int32
	merges map[[2]string]int // pair -> rank
	BOS    int32
	EOS    int32
}

// byteToRune / runeToByte is GPT-2's bytes_to_unicode: printable Latin-1
// bytes stay themselves, the others are moved to U+0100 upward.
var (
	byteToRune [256]rune
	runeToByte = map[rune]byte{}
)

func init() {
	n := rune(0)
	for b := 0; b < 256; b++ {
		printable := (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
		if printable {
			byteToRune[b] = rune(b)
		} else {
			byteToRune[b] = 256 + n
			n++
		}
		runeToByte[byteToRune[b]] = byte(b)
	}
}

// FromGGUF builds the tokenizer of f.
func FromGGUF(f *gguf.File) (*Tokenizer, error) {
	if m := f.String("tokenizer.ggml.model", ""); m != "gpt2" {
		return nil, fmt.Errorf("tokenizer: model %q is not gpt2 (byte-level BPE)", m)
	}
	tokens := f.Strings("tokenizer.ggml.tokens")
	merges := f.Strings("tokenizer.ggml.merges")
	if tokens == nil || merges == nil {
		return nil, fmt.Errorf("tokenizer: missing tokens or merges")
	}
	t := &Tokenizer{
		tokens: tokens,
		ids:    make(map[string]int32, len(tokens)),
		merges: make(map[[2]string]int, len(merges)),
		BOS:    int32(f.Uint("tokenizer.ggml.bos_token_id", 1)),
		EOS:    int32(f.Uint("tokenizer.ggml.eos_token_id", 2)),
	}
	for i, s := range tokens {
		t.ids[s] = int32(i)
	}
	if v, ok := f.Get("tokenizer.ggml.token_type"); ok {
		for _, x := range v.Ints() {
			t.types = append(t.types, int32(x))
		}
	}
	for i, m := range merges {
		a, b, ok := strings.Cut(m, " ")
		if !ok {
			return nil, fmt.Errorf("tokenizer: bad merge %q", m)
		}
		t.merges[[2]string{a, b}] = i
	}
	return t, nil
}

// Len is the vocabulary size.
func (t *Tokenizer) Len() int { return len(t.tokens) }

// ID returns the id of a token by its exact text (for special tokens
// such as "<|eot_id|>"), or -1.
func (t *Tokenizer) ID(text string) int32 {
	if id, ok := t.ids[text]; ok {
		return id
	}
	return -1
}

// IsControl reports whether id is a control (special) token.
func (t *Tokenizer) IsControl(id int32) bool {
	return int(id) < len(t.types) && t.types[id] == Control
}

// Encode tokenizes text. No special tokens are added or recognised in
// the text: build prompts from ids (see ID).
func (t *Tokenizer) Encode(text string) []int32 {
	var ids []int32
	for _, piece := range PreTokenize(text) {
		var sb strings.Builder
		for i := 0; i < len(piece); i++ {
			sb.WriteRune(byteToRune[piece[i]])
		}
		for _, sym := range t.bpe(sb.String()) {
			id, ok := t.ids[sym]
			if !ok { // cannot happen with a byte-level vocabulary
				id = t.ids[string(byteToRune[0])]
			}
			ids = append(ids, id)
		}
	}
	return ids
}

// bpe merges the characters of word by merge rank until no merge applies.
func (t *Tokenizer) bpe(word string) []string {
	syms := make([]string, 0, utf8.RuneCountInString(word))
	for _, r := range word {
		syms = append(syms, string(r))
	}
	for len(syms) > 1 {
		best, at := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if rank, ok := t.merges[[2]string{syms[i], syms[i+1]}]; ok && (best < 0 || rank < best) {
				best, at = rank, i
			}
		}
		if best < 0 {
			break
		}
		syms[at] += syms[at+1]
		syms = append(syms[:at+1], syms[at+2:]...)
	}
	return syms
}

// Decode converts ids back to text. Control tokens produce nothing.
func (t *Tokenizer) Decode(ids []int32) string {
	var b []byte
	for _, id := range ids {
		b = t.appendToken(b, id)
	}
	return string(b)
}

// Piece is the text of one token (its bytes, which may be a partial UTF-8
// sequence: concatenate pieces before printing).
func (t *Tokenizer) Piece(id int32) []byte { return t.appendToken(nil, id) }

func (t *Tokenizer) appendToken(b []byte, id int32) []byte {
	if int(id) >= len(t.tokens) || t.IsControl(id) {
		return b
	}
	for _, r := range t.tokens[id] {
		if c, ok := runeToByte[r]; ok {
			b = append(b, c)
		} else {
			b = utf8.AppendRune(b, r)
		}
	}
	return b
}

// PreTokenize splits text like the llama-bpe regex
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// with the alternatives tried in that order at each position.
func PreTokenize(text string) []string {
	var out []string
	for i := 0; i < len(text); {
		n := matchAt(text[i:])
		out = append(out, text[i:i+n])
		i += n
	}
	return out
}

func isLetter(r rune) bool  { return unicode.IsLetter(r) }
func isNumber(r rune) bool  { return unicode.IsNumber(r) }
func isSpace(r rune) bool   { return unicode.IsSpace(r) }
func isNewline(r rune) bool { return r == '\r' || r == '\n' }

// matchAt returns the byte length of the first alternative matching at
// the start of s (at least 1: the last alternative or a lone rune).
func matchAt(s string) int {
	r0, w0 := utf8.DecodeRuneInString(s)

	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if r0 == '\'' {
		rest := strings.ToLower(s[1:min(len(s), 3)])
		for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if strings.HasPrefix(rest, c) {
				return 1 + len(c)
			}
		}
	}

	// [^\r\n\p{L}\p{N}]?\p{L}+
	if isLetter(r0) {
		return w0 + spanRunes(s[w0:], isLetter)
	}
	if !isNewline(r0) && !isNumber(r0) {
		if n := spanRunes(s[w0:], isLetter); n > 0 {
			return w0 + n
		}
	}

	// \p{N}{1,3}
	if isNumber(r0) {
		n, count := 0, 0
		for n < len(s) && count < 3 {
			r, w := utf8.DecodeRuneInString(s[n:])
			if !isNumber(r) {
				break
			}
			n += w
			count++
		}
		return n
	}

	//  ?[^\s\p{L}\p{N}]+[\r\n]*
	{
		start := 0
		if r0 == ' ' {
			start = 1
		}
		punct := func(r rune) bool { return !isSpace(r) && !isLetter(r) && !isNumber(r) }
		if n := spanRunes(s[start:], punct); n > 0 {
			n += start
			return n + spanRunes(s[n:], isNewline)
		}
	}

	// The whitespace alternatives: \s*[\r\n]+ | \s+(?!\S) | \s+
	if isSpace(r0) {
		n := spanRunes(s, isSpace)
		ws := s[:n]
		if k := strings.LastIndexAny(ws, "\r\n"); k >= 0 { // \s*[\r\n]+: up to the last newline
			return k + 1
		}
		if n < len(s) { // followed by \S: \s+(?!\S) leaves the last space for the next piece
			if n > 1 {
				_, wl := utf8.DecodeLastRuneInString(ws)
				return n - wl
			}
		}
		return n // \s+ (or at end of text)
	}
	return w0
}

// spanRunes is the byte length of the longest prefix of s whose runes
// satisfy f.
func spanRunes(s string, f func(rune) bool) int {
	n := 0
	for n < len(s) {
		r, w := utf8.DecodeRuneInString(s[n:])
		if !f(r) {
			break
		}
		n += w
	}
	return n
}
