# Roadmap

What is done, what comes next, and the proof each step has to deliver. The phase table in
[README.md](README.md) is the summary; this file is the working plan. Each phase ends with a
tutorial in [docs/](docs/) explaining what was built.

## Done

### Phase 0 — foundations

- `gguf/` reader (mmap) and writer, `f16/`, `quant/` host dequantization.
- First kernels (RMSNorm, Softmax, SiluMul, Add, RoPE, MatVecF16/F32) with the discipline
  that carries the whole project: every kernel has a pure-Go twin in `backend/cpu` and a
  GPU-vs-CPU test on random data.

### Phase 1 — `gollama run`: single-token (decode) inference

- Quantized matvec kernels `MatVecQ4_0`, `MatVecQ4K`, `MatVecQ6K` (dequantize on the fly,
  one warp per output row), `AttnDecode` (one block per head, GQA), RoPE with Llama 3
  frequency factors.
- `model/`: config from metadata, weight upload in file format, KV cache, forward pass;
  `TestForwardVsCPU` checks the GPU logits against a pure-Go forward pass (4e-6).
- `tokenizer/`: byte-level BPE with a hand-written llama-bpe pre-tokenizer.
- `ollama/`: run models straight from the ollama store by name.
- Verified: greedy output identical to `ollama` (temperature 0) on Llama 3.2 3B (Q4_K_M) and
  Llama 3.1 8B (Q4_0) until the first near-tie in the logits.
- Found and fixed a cuda-ir.go bug (a `cuda.Shared` emitted as `.global` after LLVM type
  unification) — the first payoff of "the compiler is ours too".

## Next

### Phase 2 — a usable generator

Goal: prompts processed in one pass, real sampling, more files run.

1. **Batched prefill.** Feed the whole prompt as a `[T][dim]` batch. The matvec kernels
   already take a batch dimension (grid.y); what is missing is `AttnPrefill`: for T query
   rows against T keys with a causal mask, one block per (head, query-row) or a tiled
   version. RoPE/RMSNorm already work on rows. Proof: `TestPrefillVsDecode` — the KV cache
   and logits after a batched prefill equal those after T decode steps; prompt tokens/s
   goes from ~55 to > 500 on the 3B.
2. **Sampling.** `sampler/`: temperature, top-k, top-p, repetition penalty, seeded RNG; an
   `Argmax` kernel so only one int leaves the GPU per token in greedy mode. Proof: with
   temperature 0 the output is unchanged; distributions match a host reference on fixed
   logits.
3. **More weight formats.** `Q8_0`, `Q5_K`, `Q3_K`, `Q2_K`, `F16`/`BF16` matvec kernels with
   `quant/` twins; `IQ*` are out of scope. Proof: the GPU-vs-CPU tests, then any ollama
   llama-family blob runs (`gollama info` on all local blobs → all types covered).
4. **Chat.** `gollama chat`: multi-turn REPL reusing the KV cache across turns, streaming
   pieces as they decode, `<|eot_id|>` handling, the ollama/HF chat template read from
   `tokenizer.chat_template` (a small Jinja subset, or hard-coded llama 3 + a flag).
5. **Streams and pipelining.** Move launches onto a non-blocking stream, pinned host
   buffers for the embedding row and the logits, overlap the host argmax with the next
   layer. Proof: decode tok/s on the 3B measurably up; tests still green.
6. **Model zoo.** Other architectures that are "llama with a twist": Mistral (sliding
   window), Qwen2 (QKV bias), Gemma (GeLU, embedding scaling, logit soft-cap), Phi-3. Each
   is a `model.Arch` switch, not a new package.

### Phase 3 — serving

Goal: many concurrent clients, tokens/s that scales with load.

1. **Paged KV cache** (`kvcache/`): fixed-size pages per sequence, a free list, the attention
   kernel reads through a page table; makes prefix sharing and eviction possible.
2. **Continuous batching** (`engine/`): a scheduler loop that admits new requests, runs one
   batched forward step per iteration (prefill chunks + all decoding sequences), and
   removes finished ones. Needs batched-by-sequence attention and per-row positions (RoPE
   already takes `pos[row]`).
3. **HTTP server** (`server/`): OpenAI-compatible `/v1/chat/completions` and `/v1/completions`
   with SSE streaming, plus ollama-compatible `/api/generate` so existing clients work.
   Proof: `hey`/`k6` with 1, 4, 16 clients — aggregate tokens/s grows; latency per token stays
   bounded.
4. **Observability**: Prometheus metrics (tokens/s, queue depth, cache occupancy), `pprof`.

### Phase 4 — speed

Goal: within 2× of llama.cpp on the same GPU, then closer.

1. **Tensor-core GEMM** for prefill (cuda-ir.go has `wmma`): dequantize a tile of weights
   into shared memory as f16, multiply with the activation tile. This is where batched
   prefill gets its real speedup.
2. **Better matvec**: 32-bit/128-bit loads instead of byte loads, several rows per warp,
   int8-quantized activations (`Q8_1` dot products as llama.cpp does) — this is what makes
   the Q4_0 8B path several times faster than today's 9.5 tok/s.
3. **Fusion**: RMSNorm + matvec, gate/up in one launch, residual add into the matvec
   epilogue; CUDA graphs to remove launch overhead for the decode step.
4. **Tiled attention** (flash-attention style): scores never materialised, online softmax,
   tiles of K/V in shared memory; the same kernel serves prefill and decode.
5. **Speculative decoding** (draft with the 1B, verify with the 8B) once batched
   verification is cheap.

Proof for every item: the kernel tests, `TestForwardVsCPU`, and a benchmark table in
`docs/` against llama.cpp on the same GPU.

### Phase 5 — training

Goal: fine-tune on the GPU with Go kernels; see [docs/05-training.md](docs/05-training.md).

1. **Autograd**: a tape of ops over device tensors; each forward kernel gets a backward
   kernel with a CPU twin and a finite-difference gradient check.
2. **Backward kernels**: matmul (two GEMMs), RMSNorm, RoPE, SiluMul, attention, softmax +
   cross-entropy fused.
3. **LoRA / QLoRA**: low-rank adapters on the attention projections, base weights stay
   quantized; AdamW on the adapter parameters only; save adapters as a GGUF (the writer
   exists) that `gollama run --lora` applies.
4. **Full fine-tune** on small models (1B) in bf16 with gradient checkpointing; then
   dataset loading, loss masking on assistant tokens, a loss curve that matches a PyTorch
   run on the same data.
5. **Export**: write the fine-tuned weights back to GGUF, quantize them (`quant/` gains the
   encoders), and run them with `gollama run`.

## Cross-cutting

- Keep `backend/cpu` complete: it is the no-GPU fallback and the oracle; a CPU-only
  `gollama run` should work (slowly) for people without NVIDIA hardware.
- Every phase updates the tutorial that covers it and the README status line.
- cuda-ir.go improvements found here go upstream with a regression test, as the shared
  memory fix did.
