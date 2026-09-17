# gollama — llama inference and training in Go, kernels included

`gollama` runs (and fine-tunes) llama-architecture models from GGUF files on NVIDIA GPUs.
There is no C or C++ anywhere in the tree: the host is Go, the **GPU kernels are Go** too,
compiled to PTX by [cuda-ir.go](https://github.com/mehdi-shokohi/cuda-ir.go), and the CUDA
driver is reached through [gocudrv](https://github.com/eitamring/gocudrv) with
`CGO_ENABLED=0`. The result is one static binary.

Every Go ML project so far had the same shape — Go on the host, someone else's C++ on the
GPU (ggml, XLA, cuBLAS). This project is what it looks like when that boundary goes away:
a kernel is a readable Go function next to its pure-Go reference implementation, and the
same language handles the model, the tokenizer, the batching scheduler and the HTTP server.

## Status

Phase 0 of the plan below is done.

| Phase | Deliverable | Proof |
|---|---|---|
| **0** ✅ | GGUF reader/writer, first kernels, CPU oracle, test harness | `make test`: every kernel vs its CPU twin on the GPU |
| 1 | Q4_K_M forward pass, greedy decoding (Llama 3.2 3B Instruct) | same greedy tokens as `ollama` (temperature 0) on a fixed prompt |
| 2 | `Q4_0`, `Q8_0`, f16 weights; other llama-family models | same check on those files |
| 3 | serving: KV cache, continuous batching, streaming HTTP | tokens/s scales with concurrent clients |
| 4 | speed: tensor-core GEMM, fused RMSNorm+matmul, tiled attention | tokens/s vs llama.cpp on the same GPU |
| 5 | training: LoRA fine-tune, then full fine-tune | gradient check vs finite differences; loss curve |

## Layout

```
gguf/        GGUF reader (mmap) and writer, ggml type table
f16/         host-side binary16 conversions
quant/       host dequantization of Q8_0/Q4_0/Q4_K/Q6_K: the oracle for the GPU dequant kernels
kernels/     the GPU kernels, in Go  ->  go generate  ->  backend/gpu/kernels.ptx
backend/cpu  pure-Go twin of every kernel: the test oracle and the no-GPU fallback
backend/gpu  loads the embedded PTX, typed launch wrappers over gocudrv
cmd/gollama  the command (info | run | serve | train)
```

Coming with the phases: `tensor/`, `tokenizer/`, `model/`, `kvcache/`, `engine/`, `server/`, `train/`.

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

Needs the cuda-ir.go toolchain only to *regenerate* kernels (`make gen`); the PTX is
committed, so building and running the command needs just Go and an NVIDIA driver.

```bash
make test          # unit tests + kernels vs CPU oracle on the GPU
make build         # ./gollama
./gollama info model.gguf
```

## License

Apache-2.0
