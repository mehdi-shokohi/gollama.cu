// Package cpu is the pure-Go reference implementation of every kernel in
// package kernels: the oracle the GPU tests compare against, and the
// fallback when there is no GPU. Each function mirrors its kernel's
// signature with slices in place of device buffers.
package cpu

import (
	"math"

	"github.com/mehdi-shokohi/gollama/f16"
	"github.com/mehdi-shokohi/gollama/gguf"
	"github.com/mehdi-shokohi/gollama/quant"
)

// RMSNorm normalises each of the rows of n elements of x by its RMS and
// scales by w.
func RMSNorm(x, w, out []float32, rows, n int, eps float32) {
	for r := 0; r < rows; r++ {
		xr, or := x[r*n:(r+1)*n], out[r*n:(r+1)*n]
		var ss float64
		for _, v := range xr {
			ss += float64(v) * float64(v)
		}
		inv := float32(1 / math.Sqrt(ss/float64(n)+float64(eps)))
		for i, v := range xr {
			or[i] = v * inv * w[i]
		}
	}
}

// Softmax applies softmax to each of the rows of n elements of x.
func Softmax(x, out []float32, rows, n int) {
	for r := 0; r < rows; r++ {
		xr, or := x[r*n:(r+1)*n], out[r*n:(r+1)*n]
		m := float32(math.Inf(-1))
		for _, v := range xr {
			m = max(m, v)
		}
		var sum float64
		for i, v := range xr {
			e := float32(math.Exp(float64(v - m)))
			or[i] = e
			sum += float64(e)
		}
		inv := float32(1 / sum)
		for i := range or {
			or[i] *= inv
		}
	}
}

// SiluMul is out[i] = silu(gate[i]) * up[i].
func SiluMul(gate, up, out []float32) {
	for i, g := range gate {
		out[i] = g / (1 + float32(math.Exp(float64(-g)))) * up[i]
	}
}

// Add is out[i] = a[i] + b[i].
func Add(a, b, out []float32) {
	for i := range a {
		out[i] = a[i] + b[i]
	}
}

// RoPE rotates the interleaved pairs of every head of every row of x
// (rows of nHeads*headDim) by its position pos[row]. freq divides the
// angle of pair j (headDim/2 factors; nil means all ones).
func RoPE(x []float32, pos []int32, freq []float32, rows, nHeads, headDim int, base float32) {
	for r := 0; r < rows; r++ {
		p := float64(pos[r])
		for h := 0; h < nHeads; h++ {
			row := x[r*nHeads*headDim+h*headDim:][:headDim]
			for j := 0; j < headDim/2; j++ {
				theta := p * math.Pow(float64(base), -2*float64(j)/float64(headDim))
				if freq != nil {
					theta /= float64(freq[j])
				}
				s, c := math.Sincos(theta)
				x0, x1 := float64(row[2*j]), float64(row[2*j+1])
				row[2*j] = float32(x0*c - x1*s)
				row[2*j+1] = float32(x0*s + x1*c)
			}
		}
	}
}

// MatVecF16 is y[b][r] = W[r] . x[b] for an F16 [rows][cols] W and
// float32 x[batch][cols].
func MatVecF16(w []f16.Half, x, y []float32, rows, cols, batch int) {
	for b := 0; b < batch; b++ {
		xb := x[b*cols:][:cols]
		for r := 0; r < rows; r++ {
			wr := w[r*cols:][:cols]
			var acc float64
			for c, v := range wr {
				acc += float64(v.Float32()) * float64(xb[c])
			}
			y[b*rows+r] = float32(acc)
		}
	}
}

// MatVecF32 is MatVecF16 for a float32 W.
func MatVecF32(w, x, y []float32, rows, cols, batch int) {
	for b := 0; b < batch; b++ {
		xb := x[b*cols:][:cols]
		for r := 0; r < rows; r++ {
			wr := w[r*cols:][:cols]
			var acc float64
			for c, v := range wr {
				acc += float64(v) * float64(xb[c])
			}
			y[b*rows+r] = float32(acc)
		}
	}
}

// MatVecQ is MatVecF32 for a block-quantized W of typ (Q4_0, Q4_K, Q6_K,
// ...): the reference for the dequantizing kernels, row by row through
// package quant.
func MatVecQ(typ gguf.Type, w []byte, x, y []float32, rows, cols, batch int) error {
	row := make([]float32, cols)
	for r := 0; r < rows; r++ {
		if err := quant.Row(typ, w, cols, r, row); err != nil {
			return err
		}
		for b := 0; b < batch; b++ {
			xb := x[b*cols:][:cols]
			var acc float64
			for c, v := range row {
				acc += float64(v) * float64(xb[c])
			}
			y[b*rows+r] = float32(acc)
		}
	}
	return nil
}

// AttnDecode is single-token attention: q is [nHeads*headDim], kcache and
// vcache are [kvLen][nKV*headDim] and out is [nHeads*headDim]; head h
// attends to KV head h/(nHeads/nKV).
func AttnDecode(q, kcache, vcache, out []float32, kvLen, nHeads, nKV, headDim int, scale float32) {
	group := nHeads / nKV
	kvDim := nKV * headDim
	scores := make([]float32, kvLen)
	for h := 0; h < nHeads; h++ {
		qh := q[h*headDim:][:headDim]
		kvh := h / group
		for t := 0; t < kvLen; t++ {
			k := kcache[t*kvDim+kvh*headDim:][:headDim]
			var dot float64
			for d, v := range qh {
				dot += float64(v) * float64(k[d])
			}
			scores[t] = float32(dot) * scale
		}
		Softmax(scores, scores, 1, kvLen)
		o := out[h*headDim:][:headDim]
		for d := range o {
			var acc float64
			for t := 0; t < kvLen; t++ {
				acc += float64(scores[t]) * float64(vcache[t*kvDim+kvh*headDim+d])
			}
			o[d] = float32(acc)
		}
	}
}
