package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"

	"github.com/eitamring/gocudrv/cuda"
	cudair "github.com/mehdi-shokohi/cuda-ir.go"

	"github.com/mehdi-shokohi/gollama.cu/backend/gpu"
	"github.com/mehdi-shokohi/gollama.cu/gguf"
	"github.com/mehdi-shokohi/gollama.cu/ollama"
)

const cudairModule = "github.com/mehdi-shokohi/cuda-ir.go"

// doctor reports the state of everything gollama needs, with a fix for
// each missing piece. Two tiers: running a model needs Go and the NVIDIA
// driver; regenerating the kernels (make gen) also needs the cuda-ir.go
// toolchain — LLVM 22, llgo/llgen, gocuda, libdevice — which -install
// sets up by running cuda-ir.go's install.sh from the module cache.
func doctor(args []string) error {
	fs := flag.NewFlagSet("doctor", flag.ExitOnError)
	install := fs.Bool("install", false, "install the kernel toolchain (LLVM 22, llgo, gocuda) with cuda-ir.go's install.sh; asks before sudo")
	yes := fs.Bool("y", false, "with -install: answer yes to every question")
	fs.Parse(args)

	if *install {
		if err := installToolchain(*yes); err != nil {
			return err
		}
		fmt.Println()
	}

	missing := 0
	report := func(good bool, name, detail, hint string) {
		mark := "ok"
		if !good {
			mark = "MISSING"
			missing++
		}
		fmt.Printf("%-8s %-14s %s\n", mark, name, detail)
		if !good && hint != "" {
			fmt.Printf("%-8s %-14s -> %s\n", "", "", hint)
		}
	}

	fmt.Println("== to run models")
	report(true, "go", runtime.Version()+" "+runtime.GOOS+"/"+runtime.GOARCH, "")
	if err := cuda.Init(); err != nil {
		report(false, "cuda driver", err.Error(), "install the NVIDIA driver (libcuda.so.1); the CUDA toolkit is not needed to run")
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
			if k, err := gpu.Load(ctx); err != nil {
				report(false, "kernels.ptx", err.Error(), "the embedded PTX did not JIT for this GPU; regenerate it with make gen")
			} else {
				report(true, "kernels.ptx", fmt.Sprintf("%d kernels loaded (embedded sm_80 PTX, JIT'ed by the driver)", k.Len()), "")
				k.Close()
			}
			if free, tot, err := ctx.MemInfo(); err == nil {
				report(free > 3<<30, "gpu memory", fmt.Sprintf("%.1f GiB free of %.1f", float64(free)/(1<<30), float64(tot)/(1<<30)),
					"the 3B needs ~2.4 GiB, the 8B ~5 GiB; stop other GPU users (ollama keeps a model loaded for 5 min)")
			}
			ctx.Close()
		}
	}
	runnable, other, storeErr := ollamaModels()
	switch {
	case storeErr != nil:
		report(false, "models", ollama.Dir()+" (no ollama store)",
			"install ollama and `ollama pull llama3.2`; set OLLAMA_MODELS if its store is elsewhere; or give `gollama run -m` a .gguf file")
	default:
		report(len(runnable) > 0, "models", fmt.Sprintf("%s: %s", ollama.Dir(), strings.Join(runnable, " ")), "`ollama pull llama3.2`")
		if len(other) > 0 {
			fmt.Printf("%-8s %-14s not llama-architecture (not yet supported): %s\n", "", "", strings.Join(other, " "))
		}
	}
	runOK := missing == 0

	fmt.Println("\n== to regenerate the kernels (make gen)")
	toolchainErr := cudair.Doctor(os.Stdout) // llgen, LLGO_ROOT, llvm-link/opt/llc, nvptx, ptxas, libdevice, libcuda
	if p, err := exec.LookPath("gocuda"); err == nil {
		fmt.Printf("%-8s %-14s %s\n", "ok", "gocuda", p)
	} else {
		toolchainErr = errors.New("gocuda not in PATH")
		fmt.Printf("%-8s %-14s not in PATH\n%-8s %-14s -> go install %s/cmd/gocuda@%s\n", "MISSING", "gocuda", "", "", cudairModule, cudairVersion())
	}

	fmt.Println()
	switch {
	case !runOK:
		return fmt.Errorf("doctor: %d problem(s) prevent running models", missing)
	case toolchainErr != nil:
		fmt.Println("models run. The kernel toolchain is incomplete (only needed to edit kernels/):")
		fmt.Println("  gollama doctor -install      installs LLVM 22, llgo + llgen, gocuda (cuda-ir.go's install.sh)")
	default:
		fmt.Println("everything is in place, including the kernel toolchain: gollama run llama3.2 -p \"Hello\"")
	}
	return nil
}

// ollamaModels lists the models of the ollama store: those gollama can run
// and the rest with their architecture.
func ollamaModels() (runnable, other []string, err error) {
	library := filepath.Join(ollama.Dir(), "manifests", "registry.ollama.ai", "library")
	entries, err := os.ReadDir(library)
	if err != nil {
		return nil, nil, err
	}
	for _, e := range entries {
		tags, _ := os.ReadDir(filepath.Join(library, e.Name()))
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
			ref = strings.TrimSuffix(ref, ":latest")
			if arch == "llama" {
				runnable = append(runnable, ref)
			} else {
				other = append(other, ref+"("+arch+")")
			}
		}
	}
	sort.Strings(runnable)
	sort.Strings(other)
	return runnable, other, nil
}

// cudairVersion is the cuda-ir.go version this binary was built with.
func cudairVersion() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, d := range info.Deps {
			if d.Path == cudairModule {
				return d.Version
			}
		}
	}
	return "latest"
}

// cudairDir is the source directory of cuda-ir.go in the module cache
// (downloaded if needed).
func cudairDir() (string, error) {
	var mod struct{ Dir string }
	if out, err := exec.Command("go", "list", "-m", "-json", cudairModule).Output(); err == nil {
		if json.Unmarshal(out, &mod) == nil && mod.Dir != "" {
			return mod.Dir, nil
		}
	}
	out, err := exec.Command("go", "mod", "download", "-json", cudairModule+"@"+cudairVersion()).Output()
	if err != nil {
		return "", fmt.Errorf("go mod download %s: %w", cudairModule, err)
	}
	if err := json.Unmarshal(out, &mod); err != nil || mod.Dir == "" {
		return "", fmt.Errorf("go mod download: no Dir in %s", out)
	}
	return mod.Dir, nil
}

// installToolchain runs cuda-ir.go's install.sh out of the Go module cache
// (go get put it there), pinned to the version in go.mod so gocuda matches
// the cuda package the kernels import. llgo is cloned by the script: its
// runtime/ is a nested module that the module zip does not contain, and
// llgen needs that tree.
func installToolchain(yes bool) error {
	dir, err := cudairDir()
	if err != nil {
		return err
	}
	version := cudairVersion()
	fmt.Printf("== installing the kernel toolchain: %s/install.sh (cuda-ir.go %s)\n", dir, version)
	script := filepath.Join(dir, "install.sh")
	args := []string{script}
	if yes {
		args = append(args, "-y")
	}
	cmd := exec.Command("bash", args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.Env = append(os.Environ(), "GOCUDA_REF="+version)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("install.sh: %w", err)
	}
	if os.Getenv("LLGO_ROOT") == "" {
		fmt.Println("\ninstall.sh added LLGO_ROOT to your shell profile; open a new shell (or `source ~/.bashrc`) before make gen")
	}
	return nil
}
