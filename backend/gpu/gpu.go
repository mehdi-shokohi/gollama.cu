// Package gpu runs the kernels of package kernels on an NVIDIA GPU through
// gocudrv (pure Go, no cgo). The PTX is embedded; `go generate ./kernels`
// regenerates it.
package gpu

import (
	"context"
	_ "embed"
	"fmt"

	"github.com/eitamring/gocudrv/cuda"

	"github.com/mehdi-shokohi/gollama.cu/f16"
	"github.com/mehdi-shokohi/gollama.cu/gguf"
	"github.com/mehdi-shokohi/gollama.cu/kernels"
)

//go:embed kernels.ptx
var ptx []byte

// PTX is the embedded kernel module.
func PTX() []byte { return ptx }

var kernelNames = []string{"RMSNorm", "Softmax", "SiluMul", "Add", "RoPE", "MatVecF16", "MatVecF32",
	"MatVecQ4_0", "MatVecQ4K", "MatVecQ6K", "AttnDecode"}

// Kernels is the loaded kernel module of one context.
type Kernels struct {
	ctx *cuda.Context
	mod *cuda.Module
	fn  map[string]*cuda.Function
}

// Load loads the embedded PTX into ctx.
func Load(ctx *cuda.Context) (*Kernels, error) {
	mod, err := ctx.LoadModule(ptx)
	if err != nil {
		return nil, fmt.Errorf("gpu: load kernels: %w", err)
	}
	k := &Kernels{ctx: ctx, mod: mod, fn: map[string]*cuda.Function{}}
	for _, name := range kernelNames {
		f, err := mod.Function(name)
		if err != nil {
			mod.Close()
			return nil, fmt.Errorf("gpu: kernel %s: %w", name, err)
		}
		k.fn[name] = f
	}
	return k, nil
}

// Close unloads the module.
func (k *Kernels) Close() error { return k.mod.Close() }

// Context is the CUDA context the kernels were loaded into.
func (k *Kernels) Context() *cuda.Context { return k.ctx }

// launch queues kernel name on s (the default stream when s is nil).
func (k *Kernels) launch(ctx context.Context, s *cuda.Stream, name string, cfg cuda.LaunchConfig, args ...cuda.KernelArg) error {
	f := k.fn[name]
	var err error
	if s == nil {
		err = f.Launch(ctx, cfg, args...)
	} else {
		err = f.LaunchOn(ctx, s, cfg, args...)
	}
	if err != nil {
		return fmt.Errorf("gpu: %s: %w", name, err)
	}
	return nil
}

// rowCfg is one block per row.
func rowCfg(rows int) cuda.LaunchConfig {
	return cuda.LaunchConfig{GridX: uint32(rows), GridY: 1, GridZ: 1, BlockX: kernels.BlockSize, BlockY: 1, BlockZ: 1}
}

// warpRowCfg is one warp per row, batch blocks in y.
func warpRowCfg(rows, batch int) cuda.LaunchConfig {
	const rowsPerBlock = kernels.BlockSize / 32
	return cuda.LaunchConfig{GridX: uint32((rows + rowsPerBlock - 1) / rowsPerBlock), GridY: uint32(batch), GridZ: 1,
		BlockX: kernels.BlockSize, BlockY: 1, BlockZ: 1}
}

func i32(v int) cuda.KernelArg { return cuda.ArgValue(int32(v)) }

// RMSNorm normalises rows rows of n of x by their RMS, scaled by w.
func (k *Kernels) RMSNorm(ctx context.Context, s *cuda.Stream, x, w, out *cuda.Buffer[float32], rows, n int, eps float32) error {
	return k.launch(ctx, s, "RMSNorm", rowCfg(rows), cuda.Arg(x), cuda.Arg(w), cuda.Arg(out), i32(n), cuda.ArgValue(eps))
}

// Softmax applies softmax to rows rows of n of x.
func (k *Kernels) Softmax(ctx context.Context, s *cuda.Stream, x, out *cuda.Buffer[float32], rows, n int) error {
	return k.launch(ctx, s, "Softmax", rowCfg(rows), cuda.Arg(x), cuda.Arg(out), i32(n))
}

// SiluMul is out = silu(gate) * up over n elements.
func (k *Kernels) SiluMul(ctx context.Context, s *cuda.Stream, gate, up, out *cuda.Buffer[float32], n int) error {
	return k.launch(ctx, s, "SiluMul", cuda.LaunchConfig1D(n, kernels.BlockSize), cuda.Arg(gate), cuda.Arg(up), cuda.Arg(out), i32(n))
}

// Add is out = a + b over n elements.
func (k *Kernels) Add(ctx context.Context, s *cuda.Stream, a, b, out *cuda.Buffer[float32], n int) error {
	return k.launch(ctx, s, "Add", cuda.LaunchConfig1D(n, kernels.BlockSize), cuda.Arg(a), cuda.Arg(b), cuda.Arg(out), i32(n))
}

// RoPE rotates the heads of rows rows of x in place by positions pos,
// with headDim/2 frequency factors freq (see Ones for models without).
func (k *Kernels) RoPE(ctx context.Context, s *cuda.Stream, x *cuda.Buffer[float32], pos *cuda.Buffer[int32], freq *cuda.Buffer[float32], rows, nHeads, headDim int, base float32) error {
	return k.launch(ctx, s, "RoPE", rowCfg(rows), cuda.Arg(x), cuda.Arg(pos), cuda.Arg(freq), i32(nHeads), i32(headDim), cuda.ArgValue(base))
}

// AttnDecode is single-token attention of nHeads query heads in q over
// the kvLen positions of kcache/vcache ([kvLen][nKV*headDim]) into out.
func (k *Kernels) AttnDecode(ctx context.Context, s *cuda.Stream, q, kcache, vcache, out *cuda.Buffer[float32], kvLen, nHeads, nKV, headDim int, scale float32) error {
	cfg := rowCfg(nHeads)
	cfg.SharedMemBytes = uint32(kvLen * 4)
	return k.launch(ctx, s, "AttnDecode", cfg, cuda.Arg(q), cuda.Arg(kcache), cuda.Arg(vcache), cuda.Arg(out),
		i32(kvLen), i32(nKV), i32(nHeads/nKV), i32(headDim), cuda.ArgValue(scale))
}

// MatVecF16 is y[b] = W x[b] for an F16 [rows][cols] W and batch vectors.
func (k *Kernels) MatVecF16(ctx context.Context, s *cuda.Stream, w *cuda.Buffer[f16.Half], x, y *cuda.Buffer[float32], rows, cols, batch int) error {
	return k.launch(ctx, s, "MatVecF16", warpRowCfg(rows, batch), cuda.Arg(w), cuda.Arg(x), cuda.Arg(y), i32(rows), i32(cols))
}

// MatVecF32 is MatVecF16 for a float32 W.
func (k *Kernels) MatVecF32(ctx context.Context, s *cuda.Stream, w, x, y *cuda.Buffer[float32], rows, cols, batch int) error {
	return k.launch(ctx, s, "MatVecF32", warpRowCfg(rows, batch), cuda.Arg(w), cuda.Arg(x), cuda.Arg(y), i32(rows), i32(cols))
}

// MatVecQ is y[b] = W x[b] for a block-quantized [rows][cols] W of typ
// held as raw ggml bytes on the device.
func (k *Kernels) MatVecQ(ctx context.Context, s *cuda.Stream, typ gguf.Type, w *cuda.Buffer[uint8], x, y *cuda.Buffer[float32], rows, cols, batch int) error {
	var name string
	switch typ {
	case gguf.Q4_0:
		name = "MatVecQ4_0"
	case gguf.Q4_K:
		name = "MatVecQ4K"
	case gguf.Q6_K:
		name = "MatVecQ6K"
	default:
		return fmt.Errorf("gpu: no matvec kernel for %s", typ)
	}
	if cols%typ.BlockSize() != 0 {
		return fmt.Errorf("gpu: %s: %d columns is not a multiple of the block (%d)", name, cols, typ.BlockSize())
	}
	return k.launch(ctx, s, name, warpRowCfg(rows, batch), cuda.Arg(w), cuda.Arg(x), cuda.Arg(y), i32(rows), i32(cols))
}

// Ones allocates n float32 ones (the RoPE factors of a model without
// rope_freqs).
func Ones(ctx *cuda.Context, n int) (*cuda.Buffer[float32], error) {
	b, err := cuda.Alloc[float32](ctx, n)
	if err != nil {
		return nil, err
	}
	if err := b.Fill(context.Background(), 1); err != nil {
		b.Close()
		return nil, err
	}
	return b, nil
}
