# 04 — The tokenizer: text ↔ token ids

The model never sees text. It sees integers from a vocabulary of 128256 entries, and the
tokenizer is the deterministic rule that splits text into those integers and glues them
back. Llama 3 uses **byte-level BPE** in the style of GPT-2/tiktoken; `tokenizer/tokenizer.go`
implements it from the tables in the GGUF file, in ~280 lines.

```
"Why is the sky blue?"  →  [10445, 374, 279, 13180, 6437, 30]
                             Why    is   the   sky   blue   ?
```

Notice the spaces: they are attached to the *following* word (`" is"`, `" the"`), so a
sentence of five words is five tokens plus the punctuation.

## Why BPE

A vocabulary of whole words cannot spell unknown words, and a vocabulary of characters
makes every sequence long. Byte-pair encoding starts from single bytes and repeatedly
merges the most frequent adjacent pair seen in training text, 280 147 times for Llama 3.
The result is a vocabulary where common words are one token, rare words are a few
sub-word pieces, and anything at all — any language, emoji, binary — can be spelled from
the 256 byte tokens. `tokenizer.ggml.merges` is that list of 280 147 merges in the order
they were learned; the order is the priority when encoding.

## Encoding, step by step

### 1. Pre-tokenization

Text is first cut into pieces that BPE is not allowed to merge across, so that `" the"` and
`"sky"` are always tokenized independently of their neighbours. The rule is a regular
expression (`tokenizer.ggml.pre = llama-bpe`):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)          contractions:  don't → don 't
|[^\r\n\p{L}\p{N}]?\p{L}+             a word with one optional leading non-letter:  " sky", "'quote"
|\p{N}{1,3}                           digits in groups of three:  1234567 → 123 456 7
| ?[^\s\p{L}\p{N}]+[\r\n]*            punctuation runs with an optional leading space:  "!!!\n"
|\s*[\r\n]+                           whitespace ending in newlines
|\s+(?!\S)                            whitespace, leaving the last space for the next word
|\s+                                  remaining whitespace
```

Go's `regexp` has no lookahead (`(?!\S)`), so `PreTokenize` implements the alternatives by
hand, trying them in this order at each position — the order matters, regex alternation
takes the first match, not the longest. `TestPreTokenize` pins the corner cases (`"  x"`
→ `" "`, `" x"`; `"\t\tx"` → `"\t"`, `"\tx"`), and the token counts were checked against
ollama's `prompt_eval_count` on multilingual text.

### 2. Bytes to "characters"

BPE merges are stored as strings, but they operate on bytes, and many bytes are not
printable. GPT-2's trick: map each of the 256 byte values to a printable Unicode
character — printable ASCII and Latin-1 stay themselves, the rest move to U+0100 and up.
That is why the vocabulary contains tokens like `Ġsky`: `Ġ` (U+0120) is byte 0x20, the
space. `byteToRune`/`runeToByte` in the code are this table, built once in `init`.

### 3. Merging

Each pre-token, as a sequence of these characters, is merged greedily by rank:

```
symbols = characters of the piece
loop:
    find the adjacent pair with the lowest merge rank (leftmost on ties)
    if none: stop
    replace the pair by its concatenation
```

`"Ġsky"` → pairs `(Ġ,s)`, `(s,k)`, `(k,y)`; the merge list says `Ġs` came earliest… and so
on until `Ġsky` is a single symbol that is in the vocabulary: id 13180. The naive O(n²) loop
in `bpe` is fine because pre-tokens are short. Every final symbol is then looked up in the
`tokens` array; single characters always exist, so nothing is ever unknown.

### 4. Decoding

`Piece(id)` reverses the character map to bytes. A token may be a *partial* UTF-8 sequence
(the 🚀 emoji is three tokens: `" \xf0\x9f"`, `"\x9a"`, `"\x80"`), so the generation loop
in `cmd/gollama/run.go` (`utf8Writer`) holds back incomplete bytes until the next token
completes them.

## Special tokens and the chat template

Ids 128000–128255 are **control tokens** (`tokenizer.ggml.token_type = 3`). They never come
out of `Encode` — text containing `<|eot_id|>` is tokenized as ordinary characters, which
is the safe behaviour for user input — and `Decode` prints nothing for them. Prompts are
built from ids directly. An instruct model expects its conversation wrapped as it was during
fine-tuning ([05](05-training.md)):

```
<|begin_of_text|>                                          128000  BOS
<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>     optional
<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n           ← the model continues here
```

`promptTokens` in `cmd/gollama/run.go` assembles this; the model's answer ends when it
emits `<|eot_id|>` (128009), which is also the file's `eos_token_id`. A detail that
matters for reproducing ollama byte for byte: the `"\n\n"` after a header is tokenized
*together* with the content that follows, as it appears in the template text, because a
content starting with a newline would otherwise split differently.

The template ollama uses for `llama3.2` always inserts a system turn
`"Cutting Knowledge Date: December 2023\n\n"`; pass it with `-system` to get identical
prompts (`gollama run -v` prints the ids; ollama's `prompt_eval_count` should match their
count).

## Trying it

```go
tk, _ := tokenizer.FromGGUF(f)
ids := tk.Encode("naïve café — 日本語のテキスト 🚀")   // 14 tokens
fmt.Println(tk.Decode(ids))                           // round-trips exactly
fmt.Println(tk.ID("<|eot_id|>"), tk.IsControl(128009)) // 128009 true
```

`TestLlama3` holds reference ids for a few strings; add yours there when extending the
pre-tokenizer.
