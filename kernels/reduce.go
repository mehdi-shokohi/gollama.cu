// Package kernels holds the GPU kernels of gollama, written in Go and
// compiled to PTX with cuda-ir.go (`go generate ./kernels`).
//
// Conventions: a "row kernel" is launched with one block per row
// (grid.x = rows, block.x = BlockSize) and loops its threads over the
// row. Every exported function that returns nothing is a kernel; the
// backend/cpu package has a pure-Go twin of each, which the tests compare
// against.
package kernels

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// BlockSize is the thread count of every row kernel.
const BlockSize = 256

const numWarps = BlockSize / 32

// scratch is the cross-warp exchange of the block reductions.
var scratch cuda.Shared[[numWarps]float32]

// warpSum reduces v across the warp; every lane gets the total.
func warpSum(v float32) float32 {
	for d := int32(16); d > 0; d >>= 1 {
		v += cuda.ShflXorF32(cuda.FullMask, v, d)
	}
	return v
}

// warpMax is warpSum with max.
func warpMax(v float32) float32 {
	for d := int32(16); d > 0; d >>= 1 {
		v = cuda.Max(v, cuda.ShflXorF32(cuda.FullMask, v, d))
	}
	return v
}

// blockSum reduces v across the block; every thread gets the total.
// It contains barriers, so all threads must call it.
func blockSum(v float32) float32 {
	s := scratch.Get()
	lane, warp := cuda.LaneID(), cuda.ThreadIdxX()/32
	v = warpSum(v)
	cuda.SyncThreads() // a previous reduction may still be read
	if lane == 0 {
		s[warp] = v
	}
	cuda.SyncThreads()
	v = 0
	if lane < numWarps {
		v = s[lane]
	}
	return warpSum(v)
}

// blockMax is blockSum with max.
func blockMax(v float32) float32 {
	s := scratch.Get()
	lane, warp := cuda.LaneID(), cuda.ThreadIdxX()/32
	v = warpMax(v)
	cuda.SyncThreads()
	if lane == 0 {
		s[warp] = v
	}
	cuda.SyncThreads()
	v = s[0]
	if lane < numWarps {
		v = s[lane]
	}
	return warpMax(v)
}
