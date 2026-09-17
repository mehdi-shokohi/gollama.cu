package kernels

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// Quantized matrix-vector products: y[b][r] = sum_c W[r][c] * x[b][c] with
// W stored in a ggml block format and dequantized on the fly. Same launch
// shape as MatVecF16: one warp per row, grid (ceil(rows/8), batch). The
// block layouts are documented in package quant, whose Dequantize is the
// reference these kernels are tested against.

// half reads the binary16 at byte offset i.
func half(w cuda.Buf[uint8], i int32) float32 {
	return cuda.Half(uint16(w.At(i)) | uint16(w.At(i+1))<<8).Float32()
}

// MatVecQ4_0: blocks of 32 elements in 18 bytes (half d, 16 nibble
// bytes), y = d*(q-8). Each lane takes whole blocks.
func MatVecQ4_0(w cuda.Buf[uint8], x, y cuda.Buf[float32], rows, cols int32) {
	r := cuda.BlockIdxX()*numWarps + cuda.ThreadIdxX()/32
	if r >= rows {
		return
	}
	b := cuda.BlockIdxY()
	xb := b * cols
	nb := cols / 32
	wr := r * nb * 18
	var acc float32
	for blk := cuda.LaneID(); blk < nb; blk += 32 {
		p := wr + blk*18
		xo := xb + blk*32
		var s float32
		for j := int32(0); j < 16; j++ {
			q := w.At(p + 2 + j)
			s += (float32(q&0xF)-8)*x.At(xo+j) + (float32(q>>4)-8)*x.At(xo+j+16)
		}
		acc += half(w, p) * s
	}
	acc = warpSum(acc)
	if cuda.LaneID() == 0 {
		y.Set(b*rows+r, acc)
	}
}

// scaleQ4K / minQ4K unpack the 6-bit scale and min of sub-block j (0..7)
// from the 12 packed scale bytes at s (quant.ScaleMinQ4K).
func scaleQ4K(w cuda.Buf[uint8], s, j int32) float32 {
	if j < 4 {
		return float32(w.At(s+j) & 63)
	}
	return float32(w.At(s+j+4)&0xF | (w.At(s+j-4)>>6)<<4)
}

func minQ4K(w cuda.Buf[uint8], s, j int32) float32 {
	if j < 4 {
		return float32(w.At(s+j+4) & 63)
	}
	return float32(w.At(s+j+4)>>4 | (w.At(s+j)>>6)<<4)
}

// MatVecQ4K: blocks of 256 elements in 144 bytes (half d, dmin; 12 scale
// bytes; 128 nibble bytes), eight sub-blocks of 32 with y = d*sc*q -
// dmin*m. Every 64-element chunk is 32 bytes: the low nibbles are the
// first 32 elements (sub-block 2*chunk), the high nibbles the next 32.
// Lane l takes elements [8l, 8l+8) of each block, which lie in one
// sub-block, so the sum factors as d*sc*sum(q*x) - dmin*m*sum(x).
func MatVecQ4K(w cuda.Buf[uint8], x, y cuda.Buf[float32], rows, cols int32) {
	r := cuda.BlockIdxX()*numWarps + cuda.ThreadIdxX()/32
	if r >= rows {
		return
	}
	b := cuda.BlockIdxY()
	xb := b * cols
	nb := cols / 256
	wr := r * nb * 144
	lane := cuda.LaneID()
	chunk, sub := lane/8, lane%8 // 64-element chunk; 8-element group in it
	shift := (sub / 4) * 4       // groups 0-3 are low nibbles, 4-7 high
	j := 2*chunk + sub/4         // the sub-block
	var acc float32
	for blk := int32(0); blk < nb; blk++ {
		p := wr + blk*144
		qo := p + 16 + chunk*32 + (sub%4)*8
		xo := xb + blk*256 + lane*8
		var sq, sx float32
		for i := int32(0); i < 8; i++ {
			xv := x.At(xo + i)
			sq += float32(w.At(qo+i)>>shift&0xF) * xv
			sx += xv
		}
		acc += half(w, p)*scaleQ4K(w, p+4, j)*sq - half(w, p+2)*minQ4K(w, p+4, j)*sx
	}
	acc = warpSum(acc)
	if lane == 0 {
		y.Set(b*rows+r, acc)
	}
}

// MatVecQ6K: blocks of 256 elements in 210 bytes (128 low-nibble bytes ql,
// 64 high-bit bytes qh, 16 int8 scales, half d), y = d*sc*(q-32) in
// sub-blocks of 16. In each 128-element half, lane l takes the elements
// l, l+32, l+64, l+96 (quant.dequantQ6K's q1..q4).
func MatVecQ6K(w cuda.Buf[uint8], x, y cuda.Buf[float32], rows, cols int32) {
	r := cuda.BlockIdxX()*numWarps + cuda.ThreadIdxX()/32
	if r >= rows {
		return
	}
	b := cuda.BlockIdxY()
	xb := b * cols
	nb := cols / 256
	wr := r * nb * 210
	lane := cuda.LaneID()
	is := lane / 16
	var acc float32
	for blk := int32(0); blk < nb; blk++ {
		p := wr + blk*210
		var s float32
		for c := int32(0); c < 2; c++ {
			ql, qh, sc := p+c*64+lane, p+128+c*32+lane, p+192+c*8+is
			xo := xb + blk*256 + c*128 + lane
			h := w.At(qh)
			q1 := float32(int8(w.At(ql)&0xF|(h&3)<<4)) - 32
			q2 := float32(int8(w.At(ql+32)&0xF|(h>>2&3)<<4)) - 32
			q3 := float32(int8(w.At(ql)>>4|(h>>4&3)<<4)) - 32
			q4 := float32(int8(w.At(ql+32)>>4|(h>>6&3)<<4)) - 32
			s += float32(int8(w.At(sc)))*q1*x.At(xo) +
				float32(int8(w.At(sc+2)))*q2*x.At(xo+32) +
				float32(int8(w.At(sc+4)))*q3*x.At(xo+64) +
				float32(int8(w.At(sc+6)))*q4*x.At(xo+96)
		}
		acc += half(w, p+208) * s
	}
	acc = warpSum(acc)
	if lane == 0 {
		y.Set(b*rows+r, acc)
	}
}
