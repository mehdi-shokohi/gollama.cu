# 03 — Kernels in Go: running the forward pass on the GPU

Every operation in [02](02-inference.md) is a Go function in `kernels/`, compiled to PTX by
[cuda-ir.go](https://github.com/mehdi-shokohi/cuda-ir.go) and launched through
[gocudrv](https://github.com/eitamring/gocudrv) with no C in between. This page explains the
GPU execution model just far enough to read those kernels, then walks through the three
that matter most: the row reductions, the dequantizing matvec, and attention.

## The execution model in five sentences

A kernel is launched as a **grid** of **blocks**; each block has 256 **threads** here
(`kernels.BlockSize`). Threads in a block can share memory (`cuda.Shared`) and synchronize
(`cuda.SyncThreads`); blocks cannot talk to each other. Threads execute in **warps** of 32
in lockstep, and lanes of a warp can exchange registers directly with shuffles
(`cuda.ShflXorF32`). The GPU hides memory latency by keeping thousands of threads in flight,
so a good kernel gives every thread something independent to do and reads memory in
contiguous runs across adjacent lanes ("coalesced"). A Go kernel reads the coordinates
from `cuda.BlockIdxX()`, `cuda.ThreadIdxX()`, `cuda.LaneID()` and computes its share of the
work.

The conventions of this repository (from `kernels/reduce.go`):

- **Row kernel**: one block per row, `grid.x = rows`; the 256 threads stride over the row's
  elements. RMSNorm, Softmax, RoPE, and attention (one "row" per head) are row kernels.
- **Warp-per-row kernel**: for matrix-vector products, one warp per output row, 8 rows per
  block, lanes stride over the columns and reduce with shuffles.
- **Elementwise kernel**: one thread per element (`cuda.GlobalIdX()`), `LaunchConfig1D`.

### How a Go function becomes a kernel

`make gen` runs `gocuda build` on `kernels/`. Every exported function that returns nothing is
a kernel; its parameters are `cuda.Buf[T]` (a raw device pointer with `At`/`Set`) and
scalars. Unexported helpers are inlined. There are no slices, maps, closures, interfaces,
goroutines or allocations on the device, and the compiler rejects them. The output
`backend/gpu/kernels.ptx` is committed and embedded, so a user needs only Go and a driver.

On the host, `backend/gpu/gpu.go` wraps each kernel in a typed method that builds the
launch configuration and passes buffers by pointer:

```go
func (k *Kernels) RMSNorm(ctx, s, x, w, out *cuda.Buffer[float32], rows, n int, eps float32) error {
	return k.launch(ctx, s, "RMSNorm", rowCfg(rows), cuda.Arg(x), cuda.Arg(w), cuda.Arg(out), i32(n), cuda.ArgValue(eps))
}
```

## Reductions: the building block of everything

RMSNorm needs the sum of squares of a row; softmax needs a max and a sum; attention needs
both. A block-wide sum with 256 threads (`kernels/reduce.go`):

```go
func warpSum(v float32) float32 {           // every lane ends with the warp's total
	for d := int32(16); d > 0; d >>= 1 {
		v += cuda.ShflXorF32(cuda.FullMask, v, d)   // add the value of the lane d away
	}
	return v
}

var scratch cuda.Shared[[numWarps]float32]    // 8 floats of shared memory per block

func blockSum(v float32) float32 {
	s := scratch.Get()
	lane, warp := cuda.LaneID(), cuda.ThreadIdxX()/32
	v = warpSum(v)                             // 256 partial sums → 8, one per warp
	cuda.SyncThreads()
	if lane == 0 { s[warp] = v }               // each warp publishes its total
	cuda.SyncThreads()
	v = 0
	if lane < numWarps { v = s[lane] }         // warp 0..7's totals into lanes 0..7
	return warpSum(v)                          // → the block total, in every thread
}
```

Five shuffle steps reduce 32 values without touching memory; the eight warp totals meet in
shared memory. `RMSNorm` in `kernels/rows.go` is then ten lines: accumulate `v*v` over the
row, `blockSum`, `Rsqrt`, scale.

**The bug this project found.** While bringing up the 8B, attention produced wrong values
for a nondeterministic subset of heads. The generated PTX showed `scratch` declared as
`.global` rather than `.shared`: one 32-byte buffer for the whole grid, so the 32 blocks
(one per head) were overwriting each other's partial sums. The cause was in cuda-ir.go:
it recognised `cuda.Shared` variables by their type name in the linked IR, and `llvm-link`
had merged `cuda.Shared[[8]float32]` with `cuda.FragC` — two different Go types with the
same layout `{ [8 x float] }`. Tests with 7 rows never lost the race; 32 concurrent heads
did. The fix (cuda-ir.go `8aed2e0`) records the variables per package before linking;
`TestRMSNorm` now runs 1024 rows and `TestAttnDecode` the 32-head shape. Lesson: test
kernels under contention, and read the PTX when a reduction misbehaves —
`grep '\.shared' backend/gpu/kernels.ptx`.

## Dequantizing matrix-vector products

`y = W·x` with `W` stored as 4-bit blocks is 95 % of the work of a decode step. The kernel
never materialises the dequantized matrix: each lane reads a few bytes of a block, expands
them to floats in registers, multiplies with the matching `x` values, and accumulates
(`kernels/quantmv.go`). One warp per output row; a row of the 3B's `attn_q` is 3072
weights = 12 Q4_K blocks, so each lane handles 8 weights of every block:

```go
func MatVecQ4K(w cuda.Buf[uint8], x, y cuda.Buf[float32], rows, cols int32) {
	r := cuda.BlockIdxX()*numWarps + cuda.ThreadIdxX()/32     // this warp's row
	lane := cuda.LaneID()
	chunk, sub := lane/8, lane%8         // which 64-weight chunk, which 8-weight group
	shift := (sub / 4) * 4               // low or high nibbles
	j := 2*chunk + sub/4                 // the sub-block whose scale/min apply
	var acc float32
	for blk := int32(0); blk < cols/256; blk++ {
		p := wr + blk*144                                       // block start (bytes)
		qo := p + 16 + chunk*32 + (sub%4)*8                     // this lane's 8 bytes
		var sq, sx float32
		for i := int32(0); i < 8; i++ {
			xv := x.At(xo + i)
			sq += float32(w.At(qo+i)>>shift&0xF) * xv           // Σ q·x
			sx += xv                                             // Σ x
		}
		acc += half(w, p)*scaleQ4K(w, p+4, j)*sq - half(w, p+2)*minQ4K(w, p+4, j)*sx
	}
	acc = warpSum(acc)
	if lane == 0 { y.Set(b*rows+r, acc) }
}
```

The algebra is the reason it is cheap: within a sub-block `w = d·sc·q − dmin·m`, so
`Σ w·x = d·sc·Σ q·x − dmin·m·Σ x` — the scales are applied once per 8 weights rather than
per weight. `Q6_K` gives each lane the four positions its bytes encode (the interleaving
described in [01](01-gguf.md#quantization-how-3-billion-weights-fit-in-187-gib)), and
`Q4_0` gives each lane whole 18-byte blocks. All three are exactly the arithmetic of
`quant.Dequantize`, restructured for 32 lanes.

The launch shape is `grid = (ceil(rows/8), batch)`: the `batch` dimension multiplies
several input vectors by the same weights, which Phase 2's prefill will use to read the
weights once for a whole prompt.

Correctness is checked three ways: `TestMatVecQuant` against `cpu.MatVecQ` on random
blocks (with several blocks per lane, because a one-block test cannot catch a wrong
stride), `model.TestLinearVsCPU` on the real weight matrices of both models, and
`TestForwardVsCPU` end to end.

## Attention for one token

`kernels/attention.go` — one block per query head, scores in dynamic shared memory sized
by the launch (`SharedMemBytes = kvLen*4`):

```
phase 1  warp w takes positions t = w, w+8, w+16, …:
             lanes stride over the 128 dims of q_h·k_t, warpSum, lane 0 stores scores[t]
         SyncThreads
phase 2  blockMax over scores → m;  scores[t] = exp(scores[t] − m);  blockSum → inv = 1/Σ
phase 3  thread d (< 128):  out[d] = inv · Σ_t scores[t] · V[t][d]
```

Phase 1 reads K rows coalesced (adjacent lanes, adjacent floats); phase 3 reads V
coalesced across `d`. The softmax is done in place in shared memory, and the barriers
inside `blockSum` double as the fence that makes every thread's `scores` writes visible to
the others. For long contexts this kernel is limited by the single block per head; Phase 4's
tiled version splits the positions across blocks with an online softmax, the idea behind
flash attention.

## The rest

- `RoPE` rotates pairs in place, one block per row, one thread per pair; `Sincos` and `Pow`
  come from libdevice, which cuda-ir.go links in.
- `Softmax` is the row version of phase 2 (tested up to a 32000-wide row).
- `SiluMul` and `Add` are one thread per element.
- `MatVecF16`/`MatVecF32` are the unquantized matvecs, used for tests and F32 tensors.

## The CPU twin discipline

`backend/cpu/cpu.go` has a plain-Go function for every kernel with the same signature over
slices, written in the most obvious way (float64 accumulators, nested loops). It serves
three purposes: it is the specification a reader can understand in one pass; it is the
oracle of `backend/gpu/gpu_test.go` (`assertClose` on random data with tolerances that
reflect float32 vs float64 accumulation, 1e-4 for 2048-term sums); and it is the no-GPU
fallback that Phase 2 turns into a complete CPU runtime. A kernel does not land without its
twin and its test.

## Reading the PTX

`backend/gpu/kernels.ptx` is human-readable. Useful things to look for:

```
.visible .entry MatVecQ4K(              the kernel and its parameter list
.shared .align 16 .b8 …scratch[32];    static shared memory (must not be .global!)
                                       (names are the mangled Go path: github_com_mehdi_shokohi_gollama_cu_kernels_…)
.extern .shared … scores[];            dynamic shared memory, sized at launch
shfl.sync.bfly.b32                     a warp shuffle (warpSum)
bar.sync 0;                            cuda.SyncThreads
ld.global.nc / ld.shared               where a load is served from
```

`gocuda build -keep` leaves the intermediate `.ll` files next to the output, which is how
the type-unification bug was traced from PTX back to `llvm-link`.
