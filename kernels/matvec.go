package kernels

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// MatVecF16: y[b][r] = sum_c W[r][c] * x[b][c] for an F16 weight matrix
// W of [rows][cols], float32 vectors x[batch][cols]. One warp per output
// row: lanes stride over the columns and shuffle-reduce the dot product.
// Launch grid (ceil(rows/8), batch), block BlockSize. This is the decode
// (single-token) path; prefill will use a tensor-core GEMM.
func MatVecF16(w cuda.Buf[cuda.Half], x, y cuda.Buf[float32], rows, cols int32) {
	r := cuda.BlockIdxX()*numWarps + cuda.ThreadIdxX()/32
	if r >= rows {
		return
	}
	b := cuda.BlockIdxY()
	xb := b * cols
	wr := r * cols
	var acc float32
	for c := cuda.LaneID(); c < cols; c += 32 {
		acc += w.At(wr+c).Float32() * x.At(xb+c)
	}
	acc = warpSum(acc)
	if cuda.LaneID() == 0 {
		y.Set(b*rows+r, acc)
	}
}

// MatVecF32 is MatVecF16 for a float32 weight matrix (norm-free small
// matrices, tests).
func MatVecF32(w, x, y cuda.Buf[float32], rows, cols int32) {
	r := cuda.BlockIdxX()*numWarps + cuda.ThreadIdxX()/32
	if r >= rows {
		return
	}
	b := cuda.BlockIdxY()
	xb := b * cols
	wr := r * cols
	var acc float32
	for c := cuda.LaneID(); c < cols; c += 32 {
		acc += w.At(wr+c) * x.At(xb+c)
	}
	acc = warpSum(acc)
	if cuda.LaneID() == 0 {
		y.Set(b*rows+r, acc)
	}
}
