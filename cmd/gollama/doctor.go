package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/eitamring/gocudrv/cuda"

	"github.com/mehdi-shokohi/gollama.cu/backend/gpu"
	"github.com/mehdi-shokohi/gollama.cu/gguf"
	"github.com/mehdi-shokohi/gollama.cu/ollama"
)

// doctor reports the state of everything gollama needs, with a fix for
// each missing piece, and exits non-zero if running a model is impossible.
// Running needs only Go and the NVIDIA driver; regenerating the kernels
// (make gen) also needs the cuda-ir.go toolchain, which is reported as
// optional.
func doctor(args []string) error {
	if len(args) != 0 {
		usage()
	}
	missing := 0
	report := func(good bool, name, detail, hint string) {
		mark := "ok  "
		if !good {
			mark = "MISSING"
			missing++
		}
		fmt.Printf("%-8s %-16s %s\n", mark, name, detail)
		if !good && hint != "" {
			fmt.Printf("%-8s %-16s   → %s\n", "", "", hint)
		}
	}
	optional := func(good bool, name, detail, hint string) {
		mark := "ok  "
		if !good {
			mark = "opt "
		}
		fmt.Printf("%-8s %-16s %s\n", mark, name, detail)
		if !good && hint != "" {
			fmt.Printf("%-8s %-16s   → %s\n", "", "", hint)
		}
	}

	fmt.Println("gollama doctor — to run models:")
	report(true, "go", runtime.Version()+" "+runtime.GOOS+"/"+runtime.GOARCH, "")

	// The NVIDIA driver: libcuda.so.1 must load and see a device.
	if err := cuda.Init(); err != nil {
		report(false, "cuda driver", err.Error(),
			"install the NVIDIA driver (libcuda.so.1); no CUDA toolkit is needed")
	} else if dev, err := cuda.GetDevice(0); err != nil {
		report(false, "gpu", err.Error(), "no CUDA device found")
	} else {
		name, _ := dev.Name()
		total, _ := dev.TotalMemory()
		major, minor, _ := dev.ComputeCapability()
		report(true, "gpu", fmt.Sprintf("%s, %.1f GiB, sm_%d%d", name, float64(total)/(1<<30), major, minor), "")
		if ctx, err := dev.Primary(); err != nil {
			report(false, "cuda context", err.Error(), "")
		} else {
			defer ctx.Close()
			if k, err := gpu.Load(ctx); err != nil {
				report(false, "kernels.ptx", err.Error(),
					"the embedded PTX did not JIT for this GPU; run `make gen` after `make deps`")
			} else {
				k.Close()
				report(true, "kernels.ptx", fmt.Sprintf("%d kernels loaded (embedded, sm_80 PTX JIT'ed)", k.Len()), "")
			}
			if free, tot, err := ctx.MemInfo(); err == nil {
				report(free > 3<<30, "gpu memory", fmt.Sprintf("%.1f GiB free of %.1f", float64(free)/(1<<30), float64(tot)/(1<<30)),
					"the 3B needs ~2.4 GiB, the 8B ~5 GiB; stop other GPU users (ollama keeps models loaded for 5 min)")
			}
		}
	}

	// Models: the ollama store, or any GGUF the user passes.
	dir := ollama.Dir()
	manifests := filepath.Join(dir, "manifests", "registry.ollama.ai", "library")
	entries, err := os.ReadDir(manifests)
	if err != nil {
		report(false, "ollama models", dir+" (no store)",
			"install ollama and `ollama pull llama3.2`, set OLLAMA_MODELS if the store is elsewhere, or pass a .gguf path to `gollama run -m`")
	} else {
		var runnable, other []string
		for _, e := range entries {
			tags, _ := os.ReadDir(filepath.Join(manifests, e.Name()))
			for _, t := range tags {
				ref := e.Name() + ":" + t.Name()
				path, err := ollama.Resolve(ref)
				if err != nil {
					continue
				}
				f, err := gguf.Open(path)
				if err != nil {
					continue
				}
				arch := f.String("general.architecture", "?")
				f.Close()
				if arch == "llama" {
					runnable = append(runnable, strings.TrimSuffix(ref, ":latest"))
				} else {
					other = append(other, strings.TrimSuffix(ref, ":latest")+"("+arch+")")
				}
			}
		}
		sort.Strings(runnable)
		report(len(runnable) > 0, "ollama models", fmt.Sprintf("%s: %d llama-architecture: %s", dir, len(runnable), strings.Join(runnable, " ")),
			"`ollama pull llama3.2`")
		if len(other) > 0 {
			fmt.Printf("%-8s %-16s not yet supported: %s\n", "", "", strings.Join(other, " "))
		}
	}

	fmt.Println("\nto regenerate the kernels (make gen; optional):")
	if p, err := exec.LookPath("gocuda"); err != nil {
		optional(false, "gocuda", "not in PATH", "make deps (installs the cuda-ir.go toolchain next to this checkout)")
	} else {
		optional(true, "gocuda", p, "")
		out, err := exec.Command("gocuda", "doctor").CombinedOutput()
		for _, line := range strings.Split(strings.TrimRight(string(out), "\n"), "\n") {
			fmt.Printf("%-8s %-16s %s\n", "", "", line)
		}
		detail := "toolchain complete: make gen works"
		if err != nil {
			detail = "problems above (the Makefile sets LLGO_ROOT=../llgo; `make deps` installs the rest)"
		}
		optional(err == nil, "gocuda doctor", detail, "")
	}

	if missing > 0 {
		return errors.New("doctor: " + fmt.Sprint(missing) + " problem(s)")
	}
	fmt.Println("\neverything needed to run models is in place: gollama run llama3.2 -p \"Hello\"")
	return nil
}
