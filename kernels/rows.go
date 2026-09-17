package kernels

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// RMSNorm: out[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w, one block per
// row of n elements. x and out may be the same buffer.
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

// Softmax: out[r] = softmax(x[r]) over n elements, one block per row.
// x and out may be the same buffer.
func Softmax(x, out cuda.Buf[float32], n int32) {
	row := cuda.BlockIdxX() * n
	tid := cuda.ThreadIdxX()
	m := float32(-3.4028235e38)
	for i := tid; i < n; i += BlockSize {
		m = cuda.Max(m, x.At(row+i))
	}
	m = blockMax(m)
	var sum float32
	for i := tid; i < n; i += BlockSize {
		e := cuda.Exp(x.At(row+i) - m)
		out.Set(row+i, e)
		sum += e
	}
	inv := 1 / blockSum(sum)
	for i := tid; i < n; i += BlockSize {
		out.Set(row+i, out.At(row+i)*inv)
	}
}

// SiluMul: out[i] = silu(gate[i]) * up[i] over n elements (the llama MLP
// activation). Launched 1-D over n.
func SiluMul(gate, up, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i < n {
		g := gate.At(i)
		out.Set(i, g/(1+cuda.Exp(-g))*up.At(i))
	}
}

// Add: out[i] = a[i] + b[i] over n elements (the residual). Launched 1-D.
func Add(a, b, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i < n {
		out.Set(i, a.At(i)+b.At(i))
	}
}

// RoPE rotates the query/key heads of x in place: x is [rows][nHeads*headDim]
// (one block per row), the row's position is pos[row] and the pairs are
// interleaved (x[2i], x[2i+1]) — the llama layout in GGUF files, whose
// converter permutes Q/K accordingly. freq holds headDim/2 frequency
// factors (Llama 3's rope_freqs.weight; all ones when a model has none):
// theta_j = pos * base^(-2j/headDim) / freq[j].
func RoPE(x cuda.Buf[float32], pos cuda.Buf[int32], freq cuda.Buf[float32], nHeads, headDim int32, base float32) {
	r := cuda.BlockIdxX()
	p := float32(pos.At(r))
	row := r * nHeads * headDim
	half := headDim / 2
	for i := cuda.ThreadIdxX(); i < nHeads*half; i += BlockSize {
		h, j := i/half, i%half
		theta := p * cuda.Pow(base, -2*float32(j)/float32(headDim)) / freq.At(j)
		s, c := cuda.Sincos(theta)
		k := row + h*headDim + 2*j
		x0, x1 := x.At(k), x.At(k+1)
		x.Set(k, x0*c-x1*s)
		x.Set(k+1, x0*s+x1*c)
	}
}
