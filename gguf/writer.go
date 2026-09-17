package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// Writer builds a GGUF file: add metadata and tensors, then WriteTo.
// Tensors are laid out in the order they were added.
type Writer struct {
	keys      []string
	kv        map[string]Value
	tensors   []wtensor
	alignment uint32
}

type wtensor struct {
	info TensorInfo
	data []byte
}

// NewWriter returns an empty Writer with the default alignment.
func NewWriter() *Writer {
	return &Writer{kv: map[string]Value{}, alignment: defaultAlignment}
}

// Set adds a metadata value. v may be any integer, float, bool or string,
// a slice of those, or a Value.
func (w *Writer) Set(key string, v any) {
	if _, ok := w.kv[key]; !ok {
		w.keys = append(w.keys, key)
	}
	val := toValue(v)
	w.kv[key] = val
	if key == KeyAlignment {
		w.alignment = uint32(val.Uint64())
	}
}

func toValue(v any) Value {
	switch x := v.(type) {
	case Value:
		return x
	case uint8:
		return Value{Type: Uint8, V: x}
	case int8:
		return Value{Type: Int8, V: x}
	case uint16:
		return Value{Type: Uint16, V: x}
	case int16:
		return Value{Type: Int16, V: x}
	case uint32:
		return Value{Type: Uint32, V: x}
	case int32:
		return Value{Type: Int32, V: x}
	case uint64:
		return Value{Type: Uint64, V: x}
	case int64:
		return Value{Type: Int64, V: x}
	case int:
		return Value{Type: Int32, V: int32(x)}
	case uint:
		return Value{Type: Uint32, V: uint32(x)}
	case float32:
		return Value{Type: Float32, V: x}
	case float64:
		return Value{Type: Float64, V: x}
	case bool:
		return Value{Type: Bool, V: x}
	case string:
		return Value{Type: String, V: x}
	case []string:
		return arrayOf(String, len(x), func(i int) Value { return Value{Type: String, V: x[i]} })
	case []int32:
		return arrayOf(Int32, len(x), func(i int) Value { return Value{Type: Int32, V: x[i]} })
	case []uint32:
		return arrayOf(Uint32, len(x), func(i int) Value { return Value{Type: Uint32, V: x[i]} })
	case []float32:
		return arrayOf(Float32, len(x), func(i int) Value { return Value{Type: Float32, V: x[i]} })
	case []Value:
		if len(x) == 0 {
			return Value{Type: Array, Elem: String, V: x}
		}
		return Value{Type: Array, Elem: x[0].Type, V: x}
	}
	panic(fmt.Sprintf("gguf: unsupported metadata type %T", v))
}

func arrayOf(elem ValueType, n int, at func(int) Value) Value {
	a := make([]Value, n)
	for i := range a {
		a[i] = at(i)
	}
	return Value{Type: Array, Elem: elem, V: a}
}

// AddTensor adds a tensor. shape is row-major ([rows, cols]); data is its
// raw bytes in typ and must be exactly typ.RowBytes(cols)*rows long.
func (w *Writer) AddTensor(name string, shape []int, typ Type, data []byte) error {
	if !typ.Known() {
		return fmt.Errorf("gguf: unknown type %d", typ)
	}
	dims := make([]uint64, len(shape))
	n := 1
	for i, s := range shape {
		dims[len(shape)-1-i] = uint64(s)
		n *= s
	}
	if len(shape) == 0 {
		return fmt.Errorf("gguf: tensor %q has no dims", name)
	}
	inner := shape[len(shape)-1]
	if inner%typ.BlockSize() != 0 {
		return fmt.Errorf("gguf: tensor %q: inner dim %d not a multiple of %s block %d", name, inner, typ, typ.BlockSize())
	}
	if want := typ.RowBytes(inner) * (n / inner); len(data) != want {
		return fmt.Errorf("gguf: tensor %q: %d bytes, want %d", name, len(data), want)
	}
	w.tensors = append(w.tensors, wtensor{TensorInfo{Name: name, Dims: dims, Type: typ}, data})
	return nil
}

// WriteFile writes the file to path.
func (w *Writer) WriteFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if _, err := w.WriteTo(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// WriteTo writes the file.
func (w *Writer) WriteTo(out io.Writer) (int64, error) {
	// assign data offsets
	off := uint64(0)
	for i := range w.tensors {
		w.tensors[i].info.Offset = off
		off = alignUp(off+uint64(len(w.tensors[i].data)), uint64(w.alignment))
	}
	bw := &countWriter{w: bufio.NewWriter(out)}
	bw.u32(magic)
	bw.u32(3)
	bw.u64(uint64(len(w.tensors)))
	bw.u64(uint64(len(w.keys)))
	for _, k := range w.keys {
		bw.str(k)
		bw.value(w.kv[k])
	}
	for _, t := range w.tensors {
		bw.str(t.info.Name)
		bw.u32(uint32(len(t.info.Dims)))
		for _, d := range t.info.Dims {
			bw.u64(d)
		}
		bw.u32(uint32(t.info.Type))
		bw.u64(t.info.Offset)
	}
	bw.pad(w.alignment)
	for _, t := range w.tensors {
		bw.bytes(t.data)
		bw.pad(w.alignment)
	}
	if bw.err == nil {
		bw.err = bw.w.Flush()
	}
	return bw.n, bw.err
}

type countWriter struct {
	w   *bufio.Writer
	n   int64
	err error
	tmp [8]byte
}

func (c *countWriter) bytes(b []byte) {
	if c.err != nil {
		return
	}
	n, err := c.w.Write(b)
	c.n += int64(n)
	c.err = err
}

func (c *countWriter) u8(v uint8)   { c.tmp[0] = v; c.bytes(c.tmp[:1]) }
func (c *countWriter) u16(v uint16) { binary.LittleEndian.PutUint16(c.tmp[:], v); c.bytes(c.tmp[:2]) }
func (c *countWriter) u32(v uint32) { binary.LittleEndian.PutUint32(c.tmp[:], v); c.bytes(c.tmp[:4]) }
func (c *countWriter) u64(v uint64) { binary.LittleEndian.PutUint64(c.tmp[:], v); c.bytes(c.tmp[:8]) }
func (c *countWriter) str(s string) { c.u64(uint64(len(s))); c.bytes([]byte(s)) }

func (c *countWriter) pad(align uint32) {
	for c.n%int64(align) != 0 {
		c.u8(0)
	}
}

func (c *countWriter) value(v Value) {
	c.u32(uint32(v.Type))
	c.typed(v)
}

func (c *countWriter) typed(v Value) {
	switch v.Type {
	case Uint8:
		c.u8(v.V.(uint8))
	case Int8:
		c.u8(uint8(v.V.(int8)))
	case Uint16:
		c.u16(v.V.(uint16))
	case Int16:
		c.u16(uint16(v.V.(int16)))
	case Uint32:
		c.u32(v.V.(uint32))
	case Int32:
		c.u32(uint32(v.V.(int32)))
	case Float32:
		c.u32(math.Float32bits(v.V.(float32)))
	case Bool:
		if v.V.(bool) {
			c.u8(1)
		} else {
			c.u8(0)
		}
	case String:
		c.str(v.V.(string))
	case Uint64:
		c.u64(v.V.(uint64))
	case Int64:
		c.u64(uint64(v.V.(int64)))
	case Float64:
		c.u64(math.Float64bits(v.V.(float64)))
	case Array:
		a := v.V.([]Value)
		c.u32(uint32(v.Elem))
		c.u64(uint64(len(a)))
		for _, e := range a {
			c.typed(e)
		}
	default:
		c.err = fmt.Errorf("gguf: cannot write value type %d", v.Type)
	}
}
