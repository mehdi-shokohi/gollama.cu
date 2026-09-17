package gguf

import "fmt"

// Type is a ggml tensor element type (the `type` field of a tensor info).
type Type uint32

// ggml_type values, in ggml's numbering.
const (
	F32     Type = 0
	F16     Type = 1
	Q4_0    Type = 2
	Q4_1    Type = 3
	Q5_0    Type = 6
	Q5_1    Type = 7
	Q8_0    Type = 8
	Q8_1    Type = 9
	Q2_K    Type = 10
	Q3_K    Type = 11
	Q4_K    Type = 12
	Q5_K    Type = 13
	Q6_K    Type = 14
	Q8_K    Type = 15
	IQ2_XXS Type = 16
	IQ2_XS  Type = 17
	IQ3_XXS Type = 18
	IQ1_S   Type = 19
	IQ4_NL  Type = 20
	IQ3_S   Type = 21
	IQ2_S   Type = 22
	IQ4_XS  Type = 23
	I8      Type = 24
	I16     Type = 25
	I32     Type = 26
	I64     Type = 27
	F64     Type = 28
	IQ1_M   Type = 29
	BF16    Type = 30
)

type typeInfo struct {
	name      string
	blockSize int // elements per block
	typeSize  int // bytes per block
}

var typeInfos = map[Type]typeInfo{
	F32: {"F32", 1, 4}, F16: {"F16", 1, 2}, BF16: {"BF16", 1, 2}, F64: {"F64", 1, 8},
	I8: {"I8", 1, 1}, I16: {"I16", 1, 2}, I32: {"I32", 1, 4}, I64: {"I64", 1, 8},
	Q4_0: {"Q4_0", 32, 18}, Q4_1: {"Q4_1", 32, 20}, Q5_0: {"Q5_0", 32, 22}, Q5_1: {"Q5_1", 32, 24},
	Q8_0: {"Q8_0", 32, 34}, Q8_1: {"Q8_1", 32, 36},
	Q2_K: {"Q2_K", 256, 84}, Q3_K: {"Q3_K", 256, 110}, Q4_K: {"Q4_K", 256, 144},
	Q5_K: {"Q5_K", 256, 176}, Q6_K: {"Q6_K", 256, 210}, Q8_K: {"Q8_K", 256, 292},
	IQ2_XXS: {"IQ2_XXS", 256, 66}, IQ2_XS: {"IQ2_XS", 256, 74}, IQ3_XXS: {"IQ3_XXS", 256, 98},
	IQ1_S: {"IQ1_S", 256, 50}, IQ4_NL: {"IQ4_NL", 32, 18}, IQ3_S: {"IQ3_S", 256, 110},
	IQ2_S: {"IQ2_S", 256, 82}, IQ4_XS: {"IQ4_XS", 256, 136}, IQ1_M: {"IQ1_M", 256, 56},
}

func (t Type) String() string {
	if ti, ok := typeInfos[t]; ok {
		return ti.name
	}
	return fmt.Sprintf("Type(%d)", uint32(t))
}

// Known reports whether t is a type this package knows the size of.
func (t Type) Known() bool { _, ok := typeInfos[t]; return ok }

// BlockSize is the number of elements one block of t encodes (1 for
// scalar types, 32 or 256 for quantized ones).
func (t Type) BlockSize() int { return typeInfos[t].blockSize }

// TypeSize is the number of bytes one block of t takes.
func (t Type) TypeSize() int { return typeInfos[t].typeSize }

// IsQuantized reports whether t is a block-quantized type.
func (t Type) IsQuantized() bool { return typeInfos[t].blockSize > 1 }

// RowBytes is the number of bytes a row of n elements of t takes; n must
// be a multiple of BlockSize.
func (t Type) RowBytes(n int) int { return n / t.BlockSize() * t.TypeSize() }

// ValueType is the type of a metadata value.
type ValueType uint32

const (
	Uint8   ValueType = 0
	Int8    ValueType = 1
	Uint16  ValueType = 2
	Int16   ValueType = 3
	Uint32  ValueType = 4
	Int32   ValueType = 5
	Float32 ValueType = 6
	Bool    ValueType = 7
	String  ValueType = 8
	Array   ValueType = 9
	Uint64  ValueType = 10
	Int64   ValueType = 11
	Float64 ValueType = 12
)

var valueTypeNames = [...]string{"uint8", "int8", "uint16", "int16", "uint32", "int32",
	"float32", "bool", "string", "array", "uint64", "int64", "float64"}

func (v ValueType) String() string {
	if int(v) < len(valueTypeNames) {
		return valueTypeNames[v]
	}
	return fmt.Sprintf("ValueType(%d)", uint32(v))
}
