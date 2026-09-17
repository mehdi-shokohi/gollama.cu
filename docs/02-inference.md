# 02 — Inference, one token at a time

A language model is a function from a sequence of tokens to a probability distribution over
the next token. Generation is the loop "pick a token from that distribution, append it,
repeat". This page follows one token through `model.Forward` in `model/model.go` and the
pure-Go reference `cpuForward` in `model/model_test.go`, which is the same computation with
slices instead of device buffers — read them together.

## The loop around the model

```
ids = tokenize(prompt)                       # [128000, 128006, 882, ...]  (see 04)
for pos, id in ids:  logits = Forward(id, pos)   # "prefill": fill the KV cache
loop:
    next = argmax(logits)                    # greedy; sampling comes in phase 2
    if next is <|eot_id|>: stop
    print piece(next)
    logits = Forward(next, pos); pos++
```

`Forward(token, pos)` returns `[128256]` logits: one score per vocabulary entry, higher
means more likely. Softmax would turn them into probabilities; for greedy decoding the
argmax is enough. Everything the model knows about the *previous* tokens is in the KV cache
(below), which is why the loop feeds one token at a time and the position.

## The shape of the computation

For one token the model carries a single vector `x` of `dim = 3072` floats — the **residual
stream** — through 28 identical layers, then turns it into logits:

```
x = embedding[token]                                   [3072]
for each layer:
    x = x + Attention(RMSNorm(x))                        # mixes information across positions
    x = x + MLP(RMSNorm(x))                              # transforms each position on its own
logits = W_out · RMSNorm(x)                              [128256]
```

Each layer *adds* to `x` rather than replacing it (the residual connection); that is what
lets gradients flow through 28 layers during training and what makes each layer a small
correction rather than a fresh state.

### 1. Embedding

`token_embd.weight[token]` is the token's learned 3072-vector. It lives on the host side of
the mmap; `quant.Row` dequantizes it and one `CopyFrom` puts it in `m.x`. (In the 3B this
same table is the output head at the end — "tied embeddings".)

### 2. RMSNorm

```
xn[i] = x[i] / sqrt(mean(x²) + ε) · w[i]              kernels.RMSNorm, cpu.RMSNorm
```

Keeps the vector at a fixed scale so the following matrices see inputs in the range they
were trained for. `w` is the per-layer `attn_norm.weight`. It is applied to a copy (`m.xn`);
the residual `x` itself is untouched.

### 3. Projections: Q, K, V

Three matrix-vector products with `attn_q`, `attn_k`, `attn_v`:

```
q = Wq · xn      [3072] = 24 heads × 128         (query: "what am I looking for")
k = Wk · xn      [1024] =  8 heads × 128         (key:   "what do I contain")
v = Wv · xn      [1024] =  8 heads × 128         (value: "what do I pass on")
```

These are `Linear.MatVec` calls that dispatch on the weight's file type to `MatVecQ4K`,
`MatVecQ6K`, or `MatVecQ4_0` ([03](03-kernels.md)). Note K and V are a third of the size of
Q: **grouped-query attention** shares each KV head among 3 query heads (24/8), which cuts
the KV cache by 3× with little quality loss — the cache, not the compute, is what limits
long contexts on small GPUs.

### 4. RoPE: telling the model where the token is

Attention by itself is order-blind: it would score "dog bites man" like "man bites dog".
Rotary position embedding fixes that by rotating each pair of dimensions of `q` and `k` by
an angle proportional to the position:

```
for pair j of head h (j = 0..63):
    θ = pos · base^(−2j/128) / freq[j]
    (x[2j], x[2j+1]) ← rotate by θ                    kernels.RoPE, cpu.RoPE
```

Low `j` rotates fast (fine-grained, local order), high `j` slowly (coarse, long-range). The
dot product of a rotated `q` at position `p` with a rotated `k` at position `p'` depends only
on `p − p'`: relative position falls out of the geometry. `freq` is Llama 3's
`rope_freqs.weight`, which stretches the slow frequencies so the model can use a 128k
context; files without it (the 8B) get all ones. The pairs are interleaved `(x[2j], x[2j+1])`
because the GGUF converter permutes Q/K into that order for llama-architecture files.

### 5. The KV cache

The rotated `k` and the `v` of this token are appended at row `pos` of the layer's caches
(`kcache`, `vcache`, each `[maxLen][1024]` floats):

```
kcache[layer][pos] = k;   vcache[layer][pos] = v         m.kv.CopyToDeviceAt(...)
```

That is the cache's entire contract: keys and values of every position are computed once
and kept, so each new token costs one layer-pass rather than re-processing the prompt.
Cost: 2 × 1024 × 4 bytes = 8 KiB per token per layer, 224 KiB per token for the 3B,
448 MiB for a 2048-token cache (`-ctx`). The 8B needs 256 KiB per token. This linear growth
is why Phase 3's paged cache matters for serving many conversations.

### 6. Attention

For each of the 24 query heads `h` (using KV head `h / 3`):

```
scores[t] = (q_h · k_t) / sqrt(128)     for t = 0..pos          kernels.AttnDecode
p = softmax(scores)                                              cpu.AttnDecode
out_h = Σ_t p[t] · v_t                    [128]
```

The query is compared with every cached key; the softmax turns the similarities into
weights that sum to 1; the output is the weighted average of the cached values. Because we
only ever attend to positions `≤ pos`, no explicit causal mask is needed in decode — the
future is simply not in the cache yet. `sqrt(128)` keeps the dot products from growing
with the head size, which would saturate the softmax.

The 24 head outputs are concatenated into `[3072]` and mixed by `attn_output` (`Wo`), then
added to the residual: `x += Wo · attn`.

### 7. The MLP (SwiGLU)

```
xn   = RMSNorm(x) · ffn_norm
gate = W_gate · xn        [8192]
up   = W_up   · xn        [8192]
act  = silu(gate) ⊙ up    [8192]        silu(g) = g / (1 + e^{−g})     kernels.SiluMul
x   += W_down · act       [3072]
```

Two-thirds of the model's parameters are here (3 matrices of 8192×3072 per layer). The
gate/up/down structure with the SiLU gate is the "SwiGLU" variant that llama uses instead of
a plain ReLU MLP; the gating lets the layer switch pathways on and off per input.

### 8. Logits

After the last layer: `xn = RMSNorm(x) · output_norm`, then `logits = W_out · xn` — a
`[128256][3072]` matvec, the single largest operation of the forward pass (the 3B reads
its Q6_K embedding table, 323 MB, once per token for this). The logits are copied to the
host and `model.Argmax` picks the token.

## Cost model: why decode is memory-bound

Per token the 3B does about 2 × 3.2 G multiply-adds — 6.4 GFLOP, which an RTX 5060
finishes in well under a millisecond. But to do them it must **read every weight once**:
1.87 GiB per token. At the measured 52 tok/s that is 97 GB/s, a quarter of the card's
bandwidth; the rest is lost to launch overhead (15 launches and 2 copies per layer, ~480 per token, on the
default stream with synchronous copies) and to byte-wise loads in the quantized kernels.
The 8B reads 4.33 GiB per token at 9.5 tok/s (41 GB/s) because its Q4_0 kernel is the
simplest one. Both numbers are Phase 4 targets, and they are the reason every serious
runtime batches: processing 8 tokens at once reads the weights once for all 8.

The prompt is currently processed by this same loop, one token at a time ("decode-only
prefill"), which is correct but wastes that batching opportunity — Phase 2's first item.

## Where to look

| Step | GPU (`model.Forward`) | Reference (`model_test.go: cpuForward`) | Kernel / twin |
|---|---|---|---|
| embedding | `quant.Row` + `m.x.CopyFrom` | `quant.Row` | — |
| RMSNorm | `k.RMSNorm` | `cpu.RMSNorm` | `kernels/rows.go` |
| Q/K/V, Wo, MLP, head | `Linear.MatVec` → `k.MatVecQ` | `cpu.MatVecQ` (via `quant.Row`) | `kernels/quantmv.go` |
| RoPE | `k.RoPE` | `cpu.RoPE` | `kernels/rows.go` |
| cache append | `CopyToDeviceAt` | `copy` | — |
| attention | `k.AttnDecode` | `cpu.AttnDecode` | `kernels/attention.go` |
| SiLU·up, residual | `k.SiluMul`, `k.Add` | `cpu.SiluMul`, `cpu.Add` | `kernels/rows.go` |

`TestForwardVsCPU` runs both columns on the first three prompt tokens and requires the
logits to agree to 0.05 (they agree to 4e-6). When generation looks wrong, this test and a
bisection by layer count (`m.layers = m.layers[:n]`) are the way in; that is exactly how the
shared-memory bug in [03](03-kernels.md) was found.
