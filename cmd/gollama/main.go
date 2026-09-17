// Command gollama runs llama-architecture models from GGUF files on
// NVIDIA GPUs with kernels written in Go.
//
//	gollama info model.gguf|llama3.2        print the metadata and tensor table
//	gollama run llama3.2 -p "prompt" -n 64  generate greedily on the GPU
//	gollama doctor                          check the driver, the GPU, the models, the toolchain
package main

import (
	"fmt"
	"os"
	"sort"

	"github.com/mehdi-shokohi/gollama.cu/gguf"
	"github.com/mehdi-shokohi/gollama.cu/ollama"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "info":
		err = info(os.Args[2:])
	case "run":
		err = run(os.Args[2:])
	case "doctor":
		err = doctor(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "gollama:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: gollama info <model.gguf|ollama name>\n       gollama run <model> -p <prompt> [-n tokens] (see gollama run -h)\n       gollama doctor")
	os.Exit(2)
}

func info(args []string) error {
	if len(args) != 1 {
		usage()
	}
	path, err := ollama.Resolve(args[0])
	if err != nil {
		return err
	}
	f, err := gguf.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Printf("%s: GGUF v%d, alignment %d, %d keys, %d tensors\n\n", path, f.Version, f.Alignment, len(f.Keys), len(f.Tensors))
	for _, k := range f.Keys {
		v := f.KV[k]
		fmt.Printf("  %-45s %-8s %s\n", k, v.Type, truncate(v.String(), 60))
	}
	fmt.Println()

	var total int64
	byType := map[gguf.Type]int64{}
	for _, t := range f.Tensors {
		fmt.Printf("  %-40s %-8s %v\n", t.Name, t.Type, t.Shape())
		total += int64(t.Bytes())
		byType[t.Type] += int64(t.NumElems())
	}
	types := make([]gguf.Type, 0, len(byType))
	for t := range byType {
		types = append(types, t)
	}
	sort.Slice(types, func(i, j int) bool { return byType[types[i]] > byType[types[j]] })
	fmt.Printf("\n  %.2f GiB of tensor data:", float64(total)/(1<<30))
	for _, t := range types {
		fmt.Printf(" %s %.1fM", t, float64(byType[t])/1e6)
	}
	fmt.Println()
	return nil
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n-3] + "..."
	}
	return s
}
