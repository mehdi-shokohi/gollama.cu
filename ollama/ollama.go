// Package ollama finds the GGUF files of models pulled by ollama, whose
// store keeps plain GGUF blobs under a manifest per model name.
package ollama

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ModelMediaType is the manifest layer holding the GGUF file.
const ModelMediaType = "application/vnd.ollama.image.model"

type manifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
	} `json:"layers"`
}

// Dir is the model store: $OLLAMA_MODELS, else ~/.ollama/models.
func Dir() string {
	if d := os.Getenv("OLLAMA_MODELS"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".ollama", "models")
}

// Resolve turns a model reference into the path of its GGUF file. ref may
// be a GGUF file (returned as is), a manifest file, a manifest directory
// (its "latest" tag), or a model name such as "llama3.1" or
// "llama3.1:8b" looked up in Dir().
func Resolve(ref string) (string, error) {
	if st, err := os.Stat(ref); err == nil {
		if st.IsDir() {
			return fromManifest(filepath.Join(ref, "latest"))
		}
		if isGGUF(ref) {
			return ref, nil
		}
		return fromManifest(ref)
	}
	name, tag, _ := strings.Cut(ref, ":")
	if tag == "" {
		tag = "latest"
	}
	if !strings.Contains(name, "/") {
		name = "library/" + name
	}
	m := filepath.Join(Dir(), "manifests", "registry.ollama.ai", name, tag)
	if _, err := os.Stat(m); err != nil {
		return "", fmt.Errorf("ollama: %s: not a file and no manifest at %s", ref, m)
	}
	return fromManifest(m)
}

func isGGUF(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var magic [4]byte
	_, err = f.Read(magic[:])
	return err == nil && string(magic[:]) == "GGUF"
}

// fromManifest reads a manifest and returns its model blob, which lives
// in the blobs directory beside the manifests tree.
func fromManifest(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return "", fmt.Errorf("ollama: %s: %w", path, err)
	}
	for _, l := range m.Layers {
		if l.MediaType != ModelMediaType {
			continue
		}
		// .../models/manifests/registry.ollama.ai/library/name/tag
		abs, err := filepath.Abs(path)
		if err != nil {
			return "", err
		}
		root := abs
		for i := 0; i < 4; i++ {
			root = filepath.Dir(root)
		}
		if filepath.Base(root) != "manifests" {
			return "", fmt.Errorf("ollama: %s is not inside a models/manifests tree", path)
		}
		return filepath.Join(filepath.Dir(root), "blobs", strings.Replace(l.Digest, ":", "-", 1)), nil
	}
	return "", errors.New("ollama: manifest has no model layer")
}
