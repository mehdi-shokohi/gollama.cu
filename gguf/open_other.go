//go:build !unix

package gguf

import "os"

// Open reads path into memory and parses it.
func Open(path string) (*File, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(data)
}
