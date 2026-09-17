# gollama.cu tutorials

These pages explain what happens when a language model generates text, using this
repository as the worked example. Every concept points at the Go code that implements it,
so you can read the explanation and the implementation side by side. Nothing here is hidden
in a library: the file format, the tokenizer, the math of a transformer layer, and the GPU
kernels are all in this tree, in Go.

Read in order if you are new to the subject; each page stands alone otherwise.

| Page | Question it answers | Code |
|---|---|---|
| [01 — The GGUF file](01-gguf.md) | What is inside a model file? Metadata, tensors, and how 4-bit weights are stored | `gguf/`, `quant/`, `f16/` |
| [02 — Inference, one token at a time](02-inference.md) | What computation turns a token into the next token? Embeddings, RMSNorm, attention with a KV cache, RoPE, the MLP, logits | `model/`, `backend/cpu` |
| [03 — Kernels in Go](03-kernels.md) | How does that computation run on a GPU? Blocks, warps, reductions, dequantizing matvec, the attention kernel, and how a Go function becomes PTX | `kernels/`, `backend/gpu` |
| [04 — The tokenizer](04-tokenizer.md) | How does text become token ids and back? Byte-level BPE, the pre-tokenizer regex, special tokens and the chat template | `tokenizer/`, `cmd/gollama/run.go` |
| [05 — From inference to training](05-training.md) | How was the model made, and what would it take to fine-tune it here? Pre-training, SFT, preference tuning, quantization; backprop, optimizers, LoRA | Phase 5 of [ROADMAP.md](../ROADMAP.md) |

## Conventions used in the tutorials

- Shapes are written `[rows][cols]` in row-major order, the way the Go code sees them.
  GGUF stores dimensions innermost-first, so a `[out][in]` matrix appears as `dims = [in, out]`
  in the file; `gguf.TensorInfo.Shape()` reverses them for you.
- Numbers are for **Llama 3.2 3B Instruct** unless stated: 28 layers, hidden size 3072,
  24 query heads and 8 key/value heads of 128 dimensions, MLP width 8192, vocabulary 128256.
  **Llama 3.1 8B** is the same architecture with 32 layers, 4096 / 32 heads / 14336.
- "Decode" means processing one token against the cache; "prefill" means processing a whole
  prompt in one batched pass. Phase 1 does decode only; prefill is next on the roadmap.

## Running the examples

```bash
make build
./gollama info llama3.2                              # metadata + tensor table (or a .gguf path)
./gollama run llama3.2 -p "Why is the sky blue?" -v  # -v prints the config and token ids
go test ./...                                        # every kernel vs its CPU twin
```

Tests that need a model look it up in the ollama store (`$OLLAMA_MODELS` or
`~/.ollama/models`) and skip when it is absent; `GOLLAMA_LONG=1` enables the slow 8B check.
