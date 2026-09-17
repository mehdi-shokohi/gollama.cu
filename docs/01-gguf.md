# 01 — The GGUF file: what is inside a model

A "model" on disk is two things: a few hundred numbers describing the architecture and the
tokenizer, and a few billion numbers that are the learned weights. GGUF (the format of
llama.cpp and ollama) puts both in one file with a layout simple enough to memory-map:
nothing is compressed, every tensor is a contiguous byte range, and the header tells you
where. `gguf/gguf.go` reads it in ~500 lines; `gollama info` prints what it finds.

```
$ gollama info llama3.2
…/sha256-dde5aa3f…: GGUF v3, alignment 32, 30 keys, 255 tensors

  general.architecture                          string   llama
  llama.block_count                             uint32   28
  llama.embedding_length                        uint32   3072
  …
  tokenizer.ggml.tokens                         array    [128256 x string]
  tokenizer.ggml.merges                         array    [280147 x string]

  rope_freqs.weight                        F32      [64]
  token_embd.weight                        Q6_K     [128256 3072]
  blk.0.attn_norm.weight                   F32      [3072]
  blk.0.attn_q.weight                      Q4_K     [3072 3072]
  …
  1.87 GiB of tensor data: Q4_K 2422.2M Q6_K 790.4M F32 0.2M
```

## Layout

```
┌──────────────────────────────────────────────┐
│ magic "GGUF"  version u32  n_tensors u64  n_kv u64
├──────────────────────────────────────────────┤
│ n_kv × (key string, value type u32, value)    │  metadata
├──────────────────────────────────────────────┤
│ n_tensors × (name, n_dims, dims[], type, off) │  tensor infos
├──────────────────────────────────────────────┤
│ padding to `general.alignment` (default 32)   │
├──────────────────────────────────────────────┤
│ tensor data: tensor i at data_start + off_i   │  the weights
└──────────────────────────────────────────────┘
```

Everything is little-endian. Strings are a `u64` length followed by bytes (no terminator).
Arrays are an element type, a count, and the elements. That is the whole grammar;
`gguf.Parse` walks it once and keeps a `[]byte` view of the data section, so opening a
4 GB file costs nothing until a tensor's bytes are touched (`TensorInfo.Data()` returns a
sub-slice of the mapping).

Dimensions are stored **innermost first**, ggml's convention: a weight matrix that
multiplies a 3072-vector into 8192 outputs — `[8192][3072]` in row-major terms — has
`dims = [3072, 8192]`. `TensorInfo.Shape()` reverses them so the model code can say
`rows, cols := shape[0], shape[1]`.

## The metadata that defines a llama model

| Key | Llama 3.2 3B | Meaning |
|---|---|---|
| `general.architecture` | `llama` | selects the layer structure and tensor names |
| `llama.block_count` | 28 | number of transformer layers |
| `llama.embedding_length` | 3072 | hidden size `dim`: the width of the residual stream |
| `llama.feed_forward_length` | 8192 | MLP inner width |
| `llama.attention.head_count` | 24 | query heads |
| `llama.attention.head_count_kv` | 8 | key/value heads (grouped-query attention: 3 query heads share one KV head) |
| `llama.attention.key_length` | 128 | head dimension (`dim / heads` when absent) |
| `llama.rope.freq_base` | 500000 | RoPE base θ |
| `llama.attention.layer_norm_rms_epsilon` | 1e-5 | RMSNorm ε |
| `llama.context_length` | 131072 | the trained context length |
| `llama.vocab_size` | 128256 | vocabulary size (also the first dim of `token_embd`) |
| `tokenizer.ggml.model` / `.pre` | `gpt2` / `llama-bpe` | tokenizer family and pre-tokenizer regex |
| `tokenizer.ggml.tokens`, `.merges`, `.token_type` | 128256 / 280147 / 128256 | the whole tokenizer (see [04](04-tokenizer.md)) |
| `tokenizer.ggml.bos_token_id`, `.eos_token_id` | 128000, 128009 | `<\|begin_of_text\|>`, `<\|eot_id\|>` |
| `tokenizer.chat_template` | Jinja text | how to wrap a conversation in special tokens |

`model.ConfigFrom` in `model/model.go` reads exactly these keys into a `Config`.

## The tensors of one layer

For layer `i`, the file holds nine tensors named `blk.i.*`. Their shapes tell you the data
flow of a transformer layer before you read any code (details in [02](02-inference.md)):

| Tensor | Shape (3B) | Type | Role |
|---|---|---|---|
| `attn_norm.weight` | [3072] | F32 | RMSNorm scale before attention |
| `attn_q.weight` | [3072][3072] | Q4_K | 24 heads × 128 query dims from the hidden state |
| `attn_k.weight` | [1024][3072] | Q4_K | 8 heads × 128 key dims |
| `attn_v.weight` | [1024][3072] | Q6_K | 8 heads × 128 value dims |
| `attn_output.weight` | [3072][3072] | Q4_K | mixes the heads back into the hidden state |
| `ffn_norm.weight` | [3072] | F32 | RMSNorm scale before the MLP |
| `ffn_gate.weight`, `ffn_up.weight` | [8192][3072] | Q4_K | the two halves of the SwiGLU MLP |
| `ffn_down.weight` | [3072][8192] | Q6_K or Q4_K | back to the hidden size |

Outside the layers: `token_embd.weight [128256][3072]` (the embedding table, also used as
the output head in the 3B — there is no `output.weight`, the file is "tied"), `output_norm.weight`,
and `rope_freqs.weight [64]` (Llama 3's per-frequency RoPE scaling factors). The 8B has a
separate `output.weight` (Q6_K) and no `rope_freqs`.

Why are the types mixed? This file is **Q4_K_M**: the "M" recipe keeps `attn_v` and half
of the `ffn_down` matrices in the more precise Q6_K because those layers hurt quality most
when quantized. That decision was made by whoever quantized the model; the runtime just
dispatches on each tensor's type (`model.Linear.MatVec`).

## Quantization: how 3 billion weights fit in 1.87 GiB

The model was trained in 16-bit floats (6.4 GB for the 3B). Quantization stores each weight
in ~4.5 bits by grouping weights into **blocks** that share a scale. Dequantizing is
`value = scale × integer` (plus an offset for some types). The formats are defined by their
byte layouts, which `quant/quant.go` decodes on the host and `kernels/quantmv.go` decodes on
the GPU, both byte for byte:

**Q4_0** — 32 weights in 18 bytes (4.5 bits/weight)

```
┌ d: f16 ┬ 16 bytes: two 4-bit values each ─────────────────┐
│  2 B   │ q[0]&0xF → w[0] … q[15]&0xF → w[15]; q[j]>>4 → w[j+16] │
└────────┴──────────────────────────────────────────────────┘
w = d × (q − 8)            (q in 0..15, so weights are symmetric around 0)
```

**Q8_0** — 32 weights in 34 bytes: `d` then 32 signed bytes, `w = d × q`. Nearly lossless.

**Q4_K** — 256 weights in 144 bytes (4.5 bits/weight), the "K-quant" family

```
┌ d: f16 ┬ dmin: f16 ┬ 12 bytes: 8 × (6-bit scale, 6-bit min) ┬ 128 bytes of nibbles ┐
```

The 256 weights are eight sub-blocks of 32, each with its own scale `sc` and minimum `m`
packed into 6 bits, both relative to the block-wide `d` and `dmin`:
`w = d·sc × q − dmin·m`. Two levels of scaling let the format follow the local range of the
weights more closely than Q4_0 at the same size. The 6-bit fields are packed in an
irregular way that `quant.ScaleMinQ4K` (and `scaleQ4K`/`minQ4K` in the kernel) unpack.

**Q6_K** — 256 weights in 210 bytes (6.56 bits/weight)

```
┌ ql: 128 bytes (low 4 bits) ┬ qh: 64 bytes (high 2 bits) ┬ 16 × int8 scale ┬ d: f16 ┐
w = d × scale[sub-block of 16] × (q − 32)         q = low4 | high2 << 4, in 0..63
```

The order in which `ql`/`qh` bytes map to weight positions is interleaved for the
convenience of SIMD code (`quant.dequantQ6K` shows the four values a byte contributes to).
The GPU kernel gives each lane exactly those four positions, so the layout that looks
strange on paper is natural for 32 threads working side by side.

Where the number in the file name comes from: `Q4_K_M` on the 3B → 1.87 GiB / 3.2 G
parameters ≈ 5.0 bits per weight on average, because of the Q6_K tensors and the
embedding table.

## What the runtime does with all this

- Norm vectors (F32, tiny) are dequantized on the host and uploaded as `float32`.
- Quantized matrices are uploaded **as the raw bytes** (`cuda.Buffer[uint8]`); the GPU
  dequantizes inside the matvec kernel each time it reads them. Dequantizing to f16 up
  front would need 6.4 GB — more than the 8 GB card has once the KV cache is counted — and
  would also be slower: decoding is bound by memory bandwidth, so reading 4.5 bits per
  weight instead of 16 is the whole point.
- The embedding table stays on the host side of the mapping: fetching one token's row is
  `quant.Row` on 3072 values, trivial. (The 3B's tied output head is a second upload of the
  same bytes, used as a matrix.)

## Reading and writing the format yourself

`gguf/writer.go` writes files; the tests round-trip a small file. Phase 5 will use it to
save fine-tuned adapters. `f16/` converts binary16 without `math/bits` tricks so the code
stays readable; `quant.Dequantize` is deliberately the slowest, clearest version of each
decoder — it is the oracle the fast kernels are tested against, so it must be obviously
right.

Further reading: the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
and llama.cpp's `ggml-quants.c` for the encoders of the block formats.
