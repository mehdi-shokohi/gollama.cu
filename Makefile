# Running models needs Go 1.27+ and the NVIDIA driver. Editing kernels/ also needs the
# cuda-ir.go toolchain (LLVM 22, llgo, gocuda): `make doctor` reports both tiers and
# `make deps` installs the toolchain (cuda-ir.go's install.sh, run from the module cache).
ifneq (,$(wildcard ../llgo/go.mod))
LLGO_ROOT ?= $(abspath ../llgo)
else
LLGO_ROOT ?= $(HOME)/llgo
endif
export LLGO_ROOT
export CGO_ENABLED = 0

MODEL  ?= llama3.2
PROMPT ?= Why is the sky blue?

.PHONY: build run doctor test gen deps clean

build:      ## the gollama command, a static binary
	go build -o gollama ./cmd/gollama

run: build  ## generate: make run MODEL=llama3.1 PROMPT="..."
	./gollama run $(MODEL) -p "$(PROMPT)"

doctor:     ## driver, GPU, embedded kernels, models; then the kernel toolchain (llgen, LLVM, gocuda, libdevice)
	go run ./cmd/gollama doctor

test:       ## unit tests + every kernel against its CPU twin on the GPU (model tests skip without models)
	go test -count=1 ./...

gen:        ## recompile the Go kernels to PTX (needs the toolchain: make deps)
	go generate ./kernels

deps:       ## install the kernel toolchain: LLVM 22, llgo + llgen, gocuda (asks before sudo)
	go run ./cmd/gollama doctor -install

clean:
	rm -f gollama
