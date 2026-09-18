# gollama.cu — llama inference and training in Go, kernels included

`gollama.cu` runs (and fine-tunes) llama-architecture models from GGUF files on NVIDIA GPUs.
The name is the pitch — llama.cpp, but in Go, including the GPU kernels that would otherwise
be CUDA C++ (`.cu` files): here they are Go too. There is no C or C++ anywhere in the tree: the host is Go, the **GPU kernels are Go** too,
compiled to PTX by [cuda-ir.go](https://github.com/mehdi-shokohi/cuda-ir.go), and the CUDA
driver is reached through [gocudrv](https://github.com/eitamring/gocudrv) with
`CGO_ENABLED=0`. The result is one static binary.

Every Go ML project so far had the same shape — Go on the host, someone else's C++ on the
GPU (ggml, XLA, cuBLAS). This project is what it looks like when that boundary goes away:
a kernel is a readable Go function next to its pure-Go reference implementation, and the
same language handles the model, the tokenizer, the batching scheduler and the HTTP server.

## Status

Phases 0 and 1 are done: `gollama run` generates text from Llama 3.2 3B and Llama 3.1 8B
(Q4_K_M and Q4_0 GGUF files from the ollama store) and reproduces `ollama`'s greedy output.
The detailed plan for what comes next is in [ROADMAP.md](ROADMAP.md); the tutorials in
[docs/](docs/) explain what is implemented and why.

| Phase | Deliverable | Proof |
|---|---|---|
| **0** ✅ | GGUF reader/writer, first kernels, CPU oracle, test harness | `make test`: every kernel vs its CPU twin on the GPU |
| **1** ✅ | Q4_K / Q6_K / Q4_0 forward pass, KV cache, BPE tokenizer, greedy decoding | same greedy tokens as `ollama` (temperature 0): 40/40 on the 3B, 40 then a near-tie on the 8B; GPU logits vs a pure-Go forward pass to 4e-6 |
| 2 | batched prefill, sampling, more quant types, chat, a reasoning model (DeepSeek-R1 distill) | prompt tokens/s ×10; same checks on `Q8_0` / `Q5_K` / f16 files |
| 3 | serving: paged KV cache, continuous batching, streaming HTTP | tokens/s scales with concurrent clients |
| 4 | speed: tensor-core GEMM, fused kernels, tiled attention | tokens/s vs llama.cpp on the same GPU |
| 5 | training: LoRA fine-tune, then full fine-tune | gradient check vs finite differences; loss curve |

## Layout

```
gguf/        GGUF reader (mmap) and writer, ggml type table
f16/         host-side binary16 conversions
quant/       host dequantization of Q8_0/Q4_0/Q4_K/Q6_K: the oracle for the GPU dequant kernels
kernels/     the GPU kernels, in Go  ->  go generate  ->  backend/gpu/kernels.ptx
backend/cpu  pure-Go twin of every kernel: the test oracle and the no-GPU fallback
backend/gpu  loads the embedded PTX, typed launch wrappers over gocudrv
tokenizer/   byte-level BPE (llama 3 / gpt2 style) with the llama-bpe pre-tokenizer
model/       config from GGUF metadata, weight upload, KV cache, the per-token forward pass
ollama/      resolves "llama3.1" / a manifest to the GGUF blob in the ollama store
cmd/gollama  the command (info | run; serve and train come with later phases)
docs/        tutorials: GGUF, the inference pass, the kernels, the tokenizer, training
```

Coming with the phases: `sampler/`, `kvcache/`, `engine/`, `server/`, `train/`.

## A kernel

This is `kernels.RMSNorm`, the whole thing. One block per row, a block-wide
shuffle reduction, libdevice `rsqrt`:

```go
func RMSNorm(x, w, out cuda.Buf[float32], n int32, eps float32) {
	row := cuda.BlockIdxX() * n
	tid := cuda.ThreadIdxX()
	var ss float32
	for i := tid; i < n; i += BlockSize {
		v := x.At(row + i)
		ss += v * v
	}
	inv := cuda.Rsqrt(blockSum(ss)/float32(n) + eps)
	for i := tid; i < n; i += BlockSize {
		out.Set(row+i, x.At(row+i)*inv*w.At(i))
	}
}
```

Its twin in `backend/cpu` is fifteen lines of ordinary Go, and `backend/gpu`'s test runs
both on random rows and compares.

## Building

Running needs **Go 1.27+ and the NVIDIA driver** — nothing else: the kernels are committed
as PTX (`backend/gpu/kernels.ptx`) and embedded, and the driver JITs them for your GPU.

```bash
git clone https://github.com/mehdi-shokohi/gollama.cu.git && cd gollama.cu
make doctor        # driver, GPU, embedded kernels, models, kernel toolchain — with a fix for each miss
make build         # ./gollama
make test          # unit tests + kernels vs CPU oracle on the GPU
./gollama info model.gguf
./gollama run llama3.2 -p "Why is the sky blue?"          # a model name from the ollama store
./gollama run -m model.gguf -p "..." -n 64 -raw            # any llama-architecture GGUF
```

`run` accepts a GGUF path, an ollama manifest, or an ollama model name (`llama3.1`,
`llama3.2:3b`; the store is `$OLLAMA_MODELS` or `~/.ollama/models`). It applies the llama 3
chat template unless `-raw` is given; `-system` adds a system turn (ollama's `llama3.2`
template always sends `"Cutting Knowledge Date: December 2023\n\n"`, pass that to compare
outputs). `-v` prints the config and token ids.

Phase 1 is decode-only — the prompt is fed one token at a time — and runs at ~52 tok/s on
the 3B and ~9.5 tok/s on the 8B on an RTX 5060 Laptop.

### Changing the kernels

Only if you edit `kernels/`: `make gen` recompiles them to PTX with `gocuda`, which needs
the [cuda-ir.go](https://github.com/mehdi-shokohi/cuda-ir.go) toolchain. The Go part is
already in your module cache (`go.mod` pins cuda-ir.go); the rest — LLVM 22, a clone of
[llgo](https://github.com/xgo-dev/llgo) for `llgen`, the `gocuda` command — is installed by

```bash
gollama doctor -install    # = make deps; runs cuda-ir.go's install.sh from the module cache, asks before sudo
gollama doctor             # reports both tiers: what running needs, then the kernel toolchain
```

`LLGO_ROOT` must point at the llgo clone (`install.sh` adds it to your shell profile; the
Makefile also accepts `../llgo`). Commit the regenerated `kernels.ptx` with the kernel change.

## License

Apache-2.0
