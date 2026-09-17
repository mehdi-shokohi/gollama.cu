//go:build unix

package gguf

import (
	"fmt"
	"os"
	"syscall"
)

// Open memory-maps path and parses it. Tensor data is paged in on demand;
// Close unmaps it.
func Open(path string) (*File, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	st, err := fd.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() == 0 {
		return nil, fmt.Errorf("gguf: %s is empty", path)
	}
	data, err := syscall.Mmap(int(fd.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("gguf: mmap %s: %w", path, err)
	}
	f, err := Parse(data)
	if err != nil {
		syscall.Munmap(data)
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	f.close = func() error { return syscall.Munmap(data) }
	return f, nil
}
