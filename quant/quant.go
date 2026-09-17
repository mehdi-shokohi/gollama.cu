// Package quant decodes ggml block-quantized formats on the host. It is
// the reference for the GPU dequantizing kernels and is used to read the
// small tensors (norms, embeddings rows) that stay on the CPU.
//
// Layouts (little-endian; "half" is binary16):
//
//	Q8_0  32 elems / 34 B: half d; int8 q[32]              y = d*q
//	Q4_0  32 elems / 18 B: half d; u8 q[16] (nibbles)      y = d*(q-8)
//	Q4_K 256 elems / 144 B: half d, dmin; u8 scales[12]; u8 q[128]
//	      8 sub-blocks of 32 with 6-bit scale/min:        y = d*sc*q - dmin*m
//	Q6_K 256 elems / 210 B: u8 ql[128]; u8 qh[64]; int8 scales[16]; half d
//	      16 sub-blocks of 16, 6-bit quants:              y = d*sc*(q-32)
package quant

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/mehdi-shokohi/gollama.cu/f16"
	"github.com/mehdi-shokohi/gollama.cu/gguf"
)

func half(b []byte) float32 { return f16.Half(binary.LittleEndian.Uint16(b)).Float32() }

// Dequantize decodes n elements of typ from data into out (len n).
func Dequantize(typ gguf.Type, data []byte, out []float32) error {
	n := len(out)
	if n%typ.BlockSize() != 0 {
		return fmt.Errorf("quant: %d elements is not a multiple of the %s block (%d)", n, typ, typ.BlockSize())
	}
	if want := typ.RowBytes(n); len(data) < want {
		return fmt.Errorf("quant: %s: %d bytes for %d elements, want %d", typ, len(data), n, want)
	}
	switch typ {
	case gguf.F32:
		for i := range out {
			out[i] = f32(data[4*i:])
		}
	case gguf.F16:
		for i := range out {
			out[i] = half(data[2*i:])
		}
	case gguf.BF16:
		for i := range out {
			out[i] = bf16(data[2*i:])
		}
	case gguf.Q8_0:
		for b := 0; b < n/32; b++ {
			blk := data[b*34:]
			d := half(blk)
			for j := 0; j < 32; j++ {
				out[b*32+j] = d * float32(int8(blk[2+j]))
			}
		}
	case gguf.Q4_0:
		for b := 0; b < n/32; b++ {
			blk := data[b*18:]
			d := half(blk)
			for j := 0; j < 16; j++ {
				q := blk[2+j]
				out[b*32+j] = d * (float32(q&0xF) - 8)
				out[b*32+j+16] = d * (float32(q>>4) - 8)
			}
		}
	case gguf.Q4_K:
		for b := 0; b < n/256; b++ {
			dequantQ4K(data[b*144:], out[b*256:][:256])
		}
	case gguf.Q6_K:
		for b := 0; b < n/256; b++ {
			dequantQ6K(data[b*210:], out[b*256:][:256])
		}
	default:
		return fmt.Errorf("quant: %s not supported", typ)
	}
	return nil
}

func f32(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

func bf16(b []byte) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16)
}

// ScaleMinQ4K unpacks the 6-bit scale and min of sub-block j (0..7) from
// the 12 packed bytes of a Q4_K block (ggml's get_scale_min_k4).
func ScaleMinQ4K(scales []byte, j int) (sc, m uint8) {
	if j < 4 {
		return scales[j] & 63, scales[j+4] & 63
	}
	return scales[j+4]&0xF | (scales[j-4]>>6)<<4, scales[j+4]>>4 | (scales[j]>>6)<<4
}

func dequantQ4K(blk []byte, out []float32) {
	d, dmin := half(blk), half(blk[2:])
	scales, q := blk[4:16], blk[16:144]
	for chunk := 0; chunk < 4; chunk++ { // 64 elements: 32 low nibbles, then 32 high
		sc1, m1 := ScaleMinQ4K(scales, 2*chunk)
		sc2, m2 := ScaleMinQ4K(scales, 2*chunk+1)
		d1, min1 := d*float32(sc1), dmin*float32(m1)
		d2, min2 := d*float32(sc2), dmin*float32(m2)
		qs := q[chunk*32:][:32]
		y := out[chunk*64:][:64]
		for l := 0; l < 32; l++ {
			y[l] = d1*float32(qs[l]&0xF) - min1
			y[l+32] = d2*float32(qs[l]>>4) - min2
		}
	}
}

func dequantQ6K(blk []byte, out []float32) {
	ql, qh, sc := blk[0:128], blk[128:192], blk[192:208]
	d := half(blk[208:])
	for chunk := 0; chunk < 2; chunk++ { // 128 elements
		l0, h0, s0 := ql[chunk*64:], qh[chunk*32:], sc[chunk*8:]
		y := out[chunk*128:][:128]
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8(l0[l]&0xF|(h0[l]>>0&3)<<4) - 32
			q2 := int8(l0[l+32]&0xF|(h0[l]>>2&3)<<4) - 32
			q3 := int8(l0[l]>>4|(h0[l]>>4&3)<<4) - 32
			q4 := int8(l0[l+32]>>4|(h0[l]>>6&3)<<4) - 32
			y[l] = d * float32(int8(s0[is])) * float32(q1)
			y[l+32] = d * float32(int8(s0[is+2])) * float32(q2)
			y[l+64] = d * float32(int8(s0[is+4])) * float32(q3)
			y[l+96] = d * float32(int8(s0[is+6])) * float32(q4)
		}
	}
}

// Row decodes row r of a [rows][cols] tensor of typ into out (len cols).
func Row(typ gguf.Type, data []byte, cols, r int, out []float32) error {
	rb := typ.RowBytes(cols)
	return Dequantize(typ, data[r*rb:(r+1)*rb], out[:cols])
}
