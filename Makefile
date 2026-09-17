LLGO_ROOT ?= $(abspath ../llgo)
export LLGO_ROOT
export CGO_ENABLED = 0

.PHONY: gen test build

gen:        ## recompile the Go kernels to PTX (needs gocuda from cuda-ir.go on PATH)
	go generate ./kernels

test:       ## unit tests + every kernel against its CPU twin on the GPU
	go test -count=1 ./...

build:      ## the gollama command, a static binary
	go build -o gollama ./cmd/gollama
