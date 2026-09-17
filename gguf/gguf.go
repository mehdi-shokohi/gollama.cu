// Package gguf reads and writes GGUF model files (the llama.cpp / ggml
// container: metadata key-values followed by aligned tensor data).
//
// A File is parsed from a byte slice, normally a memory mapping of the
// model file, so opening a multi-gigabyte model costs nothing until a
// tensor's bytes are touched.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const (
	magic            = 0x46554747 // "GGUF" little-endian
	defaultAlignment = 32
	// KeyAlignment is the metadata key that overrides the data alignment.
	KeyAlignment = "general.alignment"
)

// File is a parsed GGUF file.
type File struct {
	Version   uint32
	Alignment uint32
	// Keys are the metadata keys in file order; KV maps them to values.
	Keys []string
	KV   map[string]Value
	// Tensors are the tensor infos in file order.
	Tensors []*TensorInfo

	byName     map[string]*TensorInfo
	data       []byte // the whole file
	dataOffset uint64 // start of the tensor data section
	close      func() error
}

// Value is a metadata value. Scalars are stored in the matching Go type
// (uint8, int8, ..., float64, bool, string); arrays as []Value with
// Elem set to the element type.
type Value struct {
	Type ValueType
	Elem ValueType // element type when Type == Array
	V    any
}

// TensorInfo describes one tensor. Dims are in ggml order: Dims[0] is the
// innermost (contiguous) dimension, so a PyTorch [out, in] weight matrix
// has Dims = [in, out].
type TensorInfo struct {
	Name   string
	Dims   []uint64
	Type   Type
	Offset uint64 // from the start of the data section

	file *File
}

// Parse parses a GGUF file held in memory. Tensor data returned by
// TensorInfo.Data aliases data.
func Parse(data []byte) (*File, error) {
	f := &File{data: data, KV: map[string]Value{}, byName: map[string]*TensorInfo{}, Alignment: defaultAlignment}
	r := &reader{buf: data}
	if r.u32() != magic {
		return nil, errors.New("gguf: bad magic")
	}
	f.Version = r.u32()
	if f.Version < 2 || f.Version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", f.Version)
	}
	nTensors, nKV := r.u64(), r.u64()
	if r.err != nil {
		return nil, r.err
	}
	if nKV > uint64(len(data)) || nTensors > uint64(len(data)) {
		return nil, errors.New("gguf: corrupt header counts")
	}
	for i := uint64(0); i < nKV; i++ {
		key := r.str()
		v, err := r.value()
		if err != nil {
			return nil, fmt.Errorf("gguf: key %q: %w", key, err)
		}
		f.Keys = append(f.Keys, key)
		f.KV[key] = v
	}
	if v, ok := f.KV[KeyAlignment]; ok {
		f.Alignment = uint32(v.Uint64())
		if f.Alignment == 0 || f.Alignment&(f.Alignment-1) != 0 {
			return nil, fmt.Errorf("gguf: bad alignment %d", f.Alignment)
		}
	}
	for i := uint64(0); i < nTensors; i++ {
		t := &TensorInfo{file: f}
		t.Name = r.str()
		nd := r.u32()
		if nd > 4 {
			return nil, fmt.Errorf("gguf: tensor %q has %d dims", t.Name, nd)
		}
		t.Dims = make([]uint64, nd)
		for j := range t.Dims {
			t.Dims[j] = r.u64()
		}
		t.Type = Type(r.u32())
		t.Offset = r.u64()
		if r.err != nil {
			return nil, r.err
		}
		if !t.Type.Known() {
			return nil, fmt.Errorf("gguf: tensor %q: unknown type %d", t.Name, t.Type)
		}
		if _, dup := f.byName[t.Name]; dup {
			return nil, fmt.Errorf("gguf: duplicate tensor %q", t.Name)
		}
		f.Tensors = append(f.Tensors, t)
		f.byName[t.Name] = t
	}
	if r.err != nil {
		return nil, r.err
	}
	f.dataOffset = alignUp(uint64(r.pos), uint64(f.Alignment))
	for _, t := range f.Tensors {
		end := f.dataOffset + t.Offset + uint64(t.Bytes())
		if t.Offset%uint64(f.Alignment) != 0 || end > uint64(len(data)) {
			return nil, fmt.Errorf("gguf: tensor %q: data out of range", t.Name)
		}
	}
	return f, nil
}

func alignUp(x, a uint64) uint64 { return (x + a - 1) &^ (a - 1) }

// Close releases the mapping made by Open. It is a no-op for Parse.
func (f *File) Close() error {
	if f.close != nil {
		err := f.close()
		f.close, f.data = nil, nil
		return err
	}
	return nil
}

// Tensor returns the tensor named name, or nil.
func (f *File) Tensor(name string) *TensorInfo { return f.byName[name] }

// DataOffset is the file offset of the tensor data section.
func (f *File) DataOffset() uint64 { return f.dataOffset }

// Get returns the metadata value for key.
func (f *File) Get(key string) (Value, bool) { v, ok := f.KV[key]; return v, ok }

// Uint returns an integer metadata value (any integer type) or def.
func (f *File) Uint(key string, def uint64) uint64 {
	if v, ok := f.KV[key]; ok && v.IsInt() {
		return v.Uint64()
	}
	return def
}

// Float returns a float metadata value or def.
func (f *File) Float(key string, def float64) float64 {
	if v, ok := f.KV[key]; ok && (v.Type == Float32 || v.Type == Float64) {
		return v.Float64()
	}
	return def
}

// String returns a string metadata value or def.
func (f *File) String(key string, def string) string {
	if v, ok := f.KV[key]; ok && v.Type == String {
		return v.V.(string)
	}
	return def
}

// Strings returns a string-array metadata value, or nil.
func (f *File) Strings(key string) []string {
	if v, ok := f.KV[key]; ok {
		return v.Strings()
	}
	return nil
}

// NumElems is the number of elements of the tensor.
func (t *TensorInfo) NumElems() int {
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	return n
}

// Bytes is the size of the tensor's data.
func (t *TensorInfo) Bytes() int {
	if len(t.Dims) == 0 {
		return 0
	}
	return t.Type.RowBytes(int(t.Dims[0])) * (t.NumElems() / int(t.Dims[0]))
}

// Shape is Dims reversed: row-major, outermost first, the way the model
// code thinks of a matrix ([rows, cols]).
func (t *TensorInfo) Shape() []int {
	s := make([]int, len(t.Dims))
	for i, d := range t.Dims {
		s[len(t.Dims)-1-i] = int(d)
	}
	return s
}

// Data is the tensor's raw bytes (aliasing the file mapping).
func (t *TensorInfo) Data() []byte {
	off := t.file.dataOffset + t.Offset
	return t.file.data[off : off+uint64(t.Bytes())]
}

func (t *TensorInfo) String() string {
	return fmt.Sprintf("%s %v %s", t.Name, t.Shape(), t.Type)
}

// IsInt reports whether v holds any integer type.
func (v Value) IsInt() bool {
	switch v.Type {
	case Uint8, Int8, Uint16, Int16, Uint32, Int32, Uint64, Int64:
		return true
	}
	return false
}

// Uint64 converts an integer or bool value.
func (v Value) Uint64() uint64 {
	switch x := v.V.(type) {
	case uint8:
		return uint64(x)
	case int8:
		return uint64(x)
	case uint16:
		return uint64(x)
	case int16:
		return uint64(x)
	case uint32:
		return uint64(x)
	case int32:
		return uint64(x)
	case uint64:
		return x
	case int64:
		return uint64(x)
	case bool:
		if x {
			return 1
		}
	}
	return 0
}

// Float64 converts a float or integer value.
func (v Value) Float64() float64 {
	switch x := v.V.(type) {
	case float32:
		return float64(x)
	case float64:
		return x
	}
	if v.IsInt() {
		return float64(int64(v.Uint64()))
	}
	return 0
}

// Strings returns the elements of a string array, or nil.
func (v Value) Strings() []string {
	if v.Type != Array || v.Elem != String {
		return nil
	}
	a := v.V.([]Value)
	s := make([]string, len(a))
	for i := range a {
		s[i] = a[i].V.(string)
	}
	return s
}

// Floats returns the elements of a numeric array as float64, or nil.
func (v Value) Floats() []float64 {
	if v.Type != Array || v.Elem == String || v.Elem == Array {
		return nil
	}
	a := v.V.([]Value)
	s := make([]float64, len(a))
	for i := range a {
		s[i] = a[i].Float64()
	}
	return s
}

// Ints returns the elements of an integer array as int64, or nil.
func (v Value) Ints() []int64 {
	if v.Type != Array {
		return nil
	}
	a := v.V.([]Value)
	s := make([]int64, len(a))
	for i := range a {
		if !a[i].IsInt() {
			return nil
		}
		s[i] = int64(a[i].Uint64())
	}
	return s
}

func (v Value) String() string {
	if v.Type == Array {
		a := v.V.([]Value)
		if len(a) > 8 {
			return fmt.Sprintf("[%d x %s]", len(a), v.Elem)
		}
		return fmt.Sprint(v.V)
	}
	return fmt.Sprint(v.V)
}

// ---- binary reader over the header

type reader struct {
	buf []byte
	pos int
	err error
}

func (r *reader) need(n int) bool {
	if r.err != nil {
		return false
	}
	if n < 0 || r.pos+n > len(r.buf) {
		r.err = errors.New("gguf: truncated header")
		return false
	}
	return true
}

func (r *reader) u8() uint8 {
	if !r.need(1) {
		return 0
	}
	v := r.buf[r.pos]
	r.pos++
	return v
}

func (r *reader) u16() uint16 {
	if !r.need(2) {
		return 0
	}
	v := binary.LittleEndian.Uint16(r.buf[r.pos:])
	r.pos += 2
	return v
}

func (r *reader) u32() uint32 {
	if !r.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.buf[r.pos:])
	r.pos += 4
	return v
}

func (r *reader) u64() uint64 {
	if !r.need(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(r.buf[r.pos:])
	r.pos += 8
	return v
}

func (r *reader) str() string {
	n := r.u64()
	if n > uint64(len(r.buf)) || !r.need(int(n)) {
		if r.err == nil {
			r.err = errors.New("gguf: bad string length")
		}
		return ""
	}
	s := string(r.buf[r.pos : r.pos+int(n)])
	r.pos += int(n)
	return s
}

func (r *reader) value() (Value, error) {
	t := ValueType(r.u32())
	v, err := r.typed(t)
	if err == nil && r.err != nil {
		err = r.err
	}
	return v, err
}

func (r *reader) typed(t ValueType) (Value, error) {
	v := Value{Type: t}
	switch t {
	case Uint8:
		v.V = r.u8()
	case Int8:
		v.V = int8(r.u8())
	case Uint16:
		v.V = r.u16()
	case Int16:
		v.V = int16(r.u16())
	case Uint32:
		v.V = r.u32()
	case Int32:
		v.V = int32(r.u32())
	case Float32:
		v.V = math.Float32frombits(r.u32())
	case Bool:
		v.V = r.u8() != 0
	case String:
		v.V = r.str()
	case Uint64:
		v.V = r.u64()
	case Int64:
		v.V = int64(r.u64())
	case Float64:
		v.V = math.Float64frombits(r.u64())
	case Array:
		v.Elem = ValueType(r.u32())
		n := r.u64()
		if n > uint64(len(r.buf)) {
			return v, errors.New("gguf: bad array length")
		}
		a := make([]Value, 0, n)
		for i := uint64(0); i < n && r.err == nil; i++ {
			e, err := r.typed(v.Elem)
			if err != nil {
				return v, err
			}
			a = append(a, e)
		}
		v.V = a
	default:
		return v, fmt.Errorf("gguf: unknown value type %d", t)
	}
	return v, nil
}
