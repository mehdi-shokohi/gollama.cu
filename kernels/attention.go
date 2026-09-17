package kernels

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// scores is the attention row of one head: kvLen floats of dynamic shared
// memory (SharedMemBytes = kvLen*4).
var scores cuda.DynShared[float32]

// AttnDecode is single-token (decode) attention: one block per query head.
// q is [nHeads*headDim] for the new token; kcache and vcache are
// [kvLen][nKV*headDim], the keys and values of every position so far
// including this one; out is [nHeads*headDim]. Head h reads KV head
// h/group (grouped-query attention). scores = softmax(q.K^T * scale), then
// out = scores.V.
func AttnDecode(q, kcache, vcache, out cuda.Buf[float32], kvLen, nKV, group, headDim int32, scale float32) {
	h := cuda.BlockIdxX()
	kvh := h / group
	kvDim := nKV * headDim
	tid := cuda.ThreadIdxX()
	lane, warp := cuda.LaneID(), tid/32
	s := scores.Buf()
	qo := h * headDim

	// One warp per position: lanes stride over the head dimension.
	for t := warp; t < kvLen; t += numWarps {
		ko := t*kvDim + kvh*headDim
		var acc float32
		for d := lane; d < headDim; d += 32 {
			acc += q.At(qo+d) * kcache.At(ko+d)
		}
		acc = warpSum(acc)
		if lane == 0 {
			s.Set(t, acc*scale)
		}
	}
	cuda.SyncThreads()

	m := float32(-3.4028235e38)
	for t := tid; t < kvLen; t += BlockSize {
		m = cuda.Max(m, s.At(t))
	}
	m = blockMax(m)
	var sum float32
	for t := tid; t < kvLen; t += BlockSize {
		e := cuda.Exp(s.At(t) - m)
		s.Set(t, e)
		sum += e
	}
	inv := 1 / blockSum(sum) // its barriers also publish the s writes above

	// One thread per output dimension, coalesced over V's rows.
	for d := tid; d < headDim; d += BlockSize {
		vo := kvh*headDim + d
		var acc float32
		for t := int32(0); t < kvLen; t++ {
			acc += s.At(t) * vcache.At(t*kvDim+vo)
		}
		out.Set(qo+d, acc*inv)
	}
}
