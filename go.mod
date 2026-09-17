module github.com/mehdi-shokohi/gollama.cu

go 1.27

replace github.com/mehdi-shokohi/cuda-ir.go => ../cuda-ir.go

require (
	github.com/eitamring/gocudrv v0.3.2
	github.com/mehdi-shokohi/cuda-ir.go v0.0.0-20260917163133-126a7f17dce8
)

require github.com/ebitengine/purego v0.10.1 // indirect
