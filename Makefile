# Running models needs Go 1.27+ and the NVIDIA driver: `make build`, `make doctor`.
# Regenerating the kernels (make gen) also needs the cuda-ir.go toolchain (LLVM 22,
# llgo, gocuda), which `make deps` installs into the directory next to this one.
CUDAIR   ?= $(abspath ../cuda-ir.go)
LLGO_ROOT ?= $(abspath ../llgo)
export LLGO_ROOT
export CGO_ENABLED = 0

MODEL  ?= llama3.2
PROMPT ?= Why is the sky blue?

.PHONY: build run doctor test gen deps clean

build:      ## the gollama command, a static binary
	go build -o gollama ./cmd/gollama

run: build  ## generate: make run MODEL=llama3.1 PROMPT="..."
	./gollama run $(MODEL) -p "$(PROMPT)"

doctor:     ## check the driver, the GPU, the embedded kernels, the model store, the kernel toolchain
	go run ./cmd/gollama doctor

test:       ## unit tests + every kernel against its CPU twin on the GPU (model tests skip without models)
	go test -count=1 ./...

gen:        ## recompile the Go kernels to PTX (needs gocuda from cuda-ir.go on PATH)
	go generate ./kernels

deps:       ## install the kernel toolchain: clones cuda-ir.go next to this checkout and runs its install.sh
	@test -d $(CUDAIR) || git clone https://github.com/mehdi-shokohi/cuda-ir.go.git $(CUDAIR)
	cd $(CUDAIR) && ./install.sh
	@echo "toolchain installed; 'go.work' lets 'make gen' compile against $(CUDAIR):"
	@test -f go.work || go work init . $(CUDAIR)
	$(MAKE) doctor

clean:
	rm -f gollama
