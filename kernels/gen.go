package kernels

// The PTX is embedded by backend/gpu; regenerate it after changing any kernel.
//
//go:generate gocuda build -o ../backend/gpu/kernels.ptx .
