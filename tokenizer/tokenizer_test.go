package tokenizer

import (
	"reflect"
	"testing"

	"github.com/mehdi-shokohi/gollama.cu/gguf"
	"github.com/mehdi-shokohi/gollama.cu/ollama"
)

func TestPreTokenize(t *testing.T) {
	cases := map[string][]string{
		"Hello, world!":                {"Hello", ",", " world", "!"},
		" 1234567 don't   \n  x":       {" ", "123", "456", "7", " don", "'t", "   \n", " ", " x"},
		"The year 2024 was great!!!\n": {"The", " year", " ", "202", "4", " was", " great", "!!!\n"},
		"  leading and trailing  ":     {" ", " leading", " and", " trailing", "  "},
		"I'LL 'quote' it":              {"I", "'LL", " '", "quote", "'", " it"},
		"a\n\n\nb":                     {"a", "\n\n\n", "b"},
		"tabs\t\tx":                    {"tabs", "\t", "\tx"},
	}
	for in, want := range cases {
		if got := PreTokenize(in); !reflect.DeepEqual(got, want) {
			t.Errorf("PreTokenize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestLlama3(t *testing.T) {
	path, err := ollama.Resolve("llama3.2")
	if err != nil {
		t.Skipf("model not on disk: %v", err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tk, err := FromGGUF(f)
	if err != nil {
		t.Fatal(err)
	}
	if tk.BOS != 128000 || tk.EOS != 128009 || tk.ID("<|eot_id|>") != 128009 || !tk.IsControl(128009) {
		t.Errorf("special tokens: bos %d eos %d eot %d", tk.BOS, tk.EOS, tk.ID("<|eot_id|>"))
	}
	// Reference ids from llama.cpp / ollama for Llama 3.
	cases := map[string][]int32{
		"Hello, world!":                {9906, 11, 1917, 0},
		"Why is the sky blue?":         {10445, 374, 279, 13180, 6437, 30},
		" 1234567 don't   \n  x":       {220, 4513, 10961, 22, 1541, 956, 5996, 220, 865},
		"The year 2024 was great!!!\n": {791, 1060, 220, 2366, 19, 574, 2294, 80395},
		"naïve café — 日本語のテキスト 🚀":      {3458, 38672, 588, 53050, 2001, 105180, 102158, 16144, 57933, 62903, 71634, 11410, 248, 222},
	}
	for in, want := range cases {
		got := tk.Encode(in)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("Encode(%q) = %v, want %v", in, got, want)
		}
		if back := tk.Decode(got); back != in {
			t.Errorf("Decode(Encode(%q)) = %q", in, back)
		}
	}
}
