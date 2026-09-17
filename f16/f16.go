// Package f16 converts between IEEE 754 binary16 (the ggml F16 / CUDA
// __half format) and float32 on the host. Kernels use cuda.Half instead.
package f16

import "math"

// Half is a binary16 value stored in its 16 bits.
type Half uint16

// From rounds x to the nearest half (ties to even), like cvt.rn.f16.f32.
func From(x float32) Half {
	b := math.Float32bits(x)
	sign := Half(b>>16) & 0x8000
	exp := int32(b>>23&0xff) - 127
	mant := b & 0x7fffff
	switch {
	case exp == 128: // inf / nan
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp > 15: // overflow
		return sign | 0x7c00
	case exp >= -14: // normal
		m := mant >> 13
		rem := mant & 0x1fff
		if rem > 0x1000 || (rem == 0x1000 && m&1 == 1) {
			m++ // carries into the exponent correctly
		}
		return sign | Half(uint32(exp+15)<<10) + Half(m)
	case exp >= -25: // subnormal
		full := mant | 0x800000
		shift := uint32(-exp - 1) // 14..25
		m := full >> shift
		rem := full & (1<<shift - 1)
		halfway := uint32(1) << (shift - 1)
		if rem > halfway || (rem == halfway && m&1 == 1) {
			m++
		}
		return sign | Half(m)
	default:
		return sign
	}
}

// Float32 widens h.
func (h Half) Float32() float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal: normalise
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}

// ToFloat32 converts a whole slice.
func ToFloat32(dst []float32, src []Half) {
	for i, h := range src {
		dst[i] = h.Float32()
	}
}

// FromFloat32 converts a whole slice.
func FromFloat32(dst []Half, src []float32) {
	for i, x := range src {
		dst[i] = From(x)
	}
}
