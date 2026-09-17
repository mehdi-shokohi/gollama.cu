# 05 — From inference to training

The previous pages describe a model being *used*. This page describes how the weights in
the GGUF file came to be, and what it would take to change them — the goal of Phase 5 of
the [roadmap](../ROADMAP.md). It is deliberately conceptual: there is no training code in
the repository yet, and the point is to see how every piece of it maps onto things that
already exist here.

## The life of a model

```
 web text, code, books            instructions + answers        human preferences
        │                                 │                            │
   1. pre-training  ───────►  2. supervised fine-tuning  ───►  3. preference tuning
   (Llama-3.2-3B)               (…-3B-Instruct)                 (still …-Instruct)
                                                                       │
                                                 4. quantization  ◄────┘
                                                 (Q4_K_M GGUF: this repo's input)
                                                         │
                                                 5. inference (phases 1–4)
```

### 1. Pre-training: next-token prediction

The objective is embarrassingly simple: given a sequence of tokens, predict the next one.
For every position `t` the model outputs logits (exactly the `[128256]` vector from
[02](02-inference.md)), and the loss is the **cross-entropy** between those logits and the
token that actually came next:

```
p = softmax(logits_t)
loss_t = −log p[token_{t+1}]
loss   = mean over all positions and all sequences in the batch
```

Everything else — grammar, facts, reasoning, translation — is a side effect of getting
this number down over ~10¹³ tokens. A 3B model at this stage is a completion engine:
give it "Why is the sky blue?" and it continues with more questions, or a forum thread.

Note what training needs that inference does not: **all positions at once** (a batched
prefill over a whole sequence, with a causal mask so position `t` cannot see `t+1`), the
loss, and the gradients of the loss with respect to every weight.

### 2. Supervised fine-tuning (SFT): teaching the format

The same objective on a small, curated dataset of conversations wrapped in the chat template
of [04](04-tokenizer.md). Two details: the loss is usually **masked** to the assistant's
tokens (the model is not trained to predict what the user says), and the special tokens
`<|start_header_id|>`, `<|eot_id|>` are how the model learns when to speak and when to stop.
After SFT, the same prompt yields "The sky appears blue because of Rayleigh scattering…"
followed by `<|eot_id|>`. This is the stage that turns `Llama-3.2-3B` into `-Instruct`.

### 3. Preference tuning: RLHF, DPO

SFT teaches the model to imitate answers; preference tuning teaches it which of two answers
is *better* according to human raters. RLHF trains a reward model on comparisons and then
optimises the policy (the LLM) with PPO against it; DPO folds the two into one
classification-like loss on (preferred, rejected) pairs, with no reward model or sampling
loop. Either way the machinery is again forward passes, a scalar loss, and gradients.

### 4. Quantization: post-training compression

The formats in [01](01-gguf.md) are applied *after* training: each block's scale is chosen
to minimise the rounding error of its weights (the K-quants also use an "importance
matrix" from calibration data to decide which weights matter). No gradients are involved.
`quant/` currently only decodes; Phase 5 adds the encoders so fine-tuned weights can be
written back.

## What training needs from this codebase

Inference computes `y = f(x; W)`. Training also computes `∂loss/∂W` for every `W` and steps
the weights downhill. Concretely, four things:

### Backpropagation: a backward kernel for every forward kernel

The chain rule walks the forward pass backwards. For each operation with inputs `a` and
output `y`, given `∂loss/∂y` (call it `dy`), produce `∂loss/∂a` and, if the op has weights,
`∂loss/∂W`:

| Forward (exists) | Backward (Phase 5) |
|---|---|
| `y = W·x` (matvec / GEMM) | `dx = Wᵀ·dy`, `dW = dy ⊗ x` — two more GEMMs; `dW` is the size of `W` |
| `RMSNorm` | `dx` from `dy`, the saved `x` and the row's RMS; `dw = Σ dy·x̂` |
| `RoPE` | rotate `dy` by `−θ` (its own inverse) |
| `SiluMul` | `dgate = dy·up·silu'(gate)`, `dup = dy·silu(gate)` |
| attention | `dq, dk, dv` from `dy`, the saved probabilities and `V`; the expensive one |
| softmax + cross-entropy | `dlogits = p − onehot(target)` — the famous one-liner |
| `Add` (residual) | `dy` flows to both inputs unchanged |

Two consequences shape the design. **Activations must be saved**: the backward of a layer
needs its forward inputs, so a training step keeps `x`, `xn`, `q`, `k`, `v`, the attention
probabilities and the MLP activations of every layer and every position — for a 2048-token
sequence on the 3B that is gigabytes, and *gradient checkpointing* (recompute instead of
store) is the standard trade. **Weights must be differentiable**: 4-bit blocks are not, so
either the weights are held in bf16 (full fine-tuning, 6.4 GB for the 3B plus gradients
plus optimizer state — too big for 8 GB) or the quantized weights are frozen and small
trainable matrices are added beside them.

### Autograd: recording the forward pass

Writing the backward pass by hand for a fixed architecture is possible (llama.cpp's
`train` did it), but a small **tape** is cleaner: every op records its inputs, output and
backward function; `loss.Backward()` replays the tape in reverse accumulating gradients.
The tape needs device tensors with shape and dtype — the `tensor/` package the README
promises — over the same `cuda.Buffer`s the model uses today.

### An optimizer

Plain SGD (`W -= lr · dW`) is rarely used for transformers; **AdamW** keeps two running
moments per parameter (`m`, `v`), scales each step by `m/√v`, and decays the weights
separately. That is two extra copies of every trainable parameter in float32 — the reason
"7B full fine-tune" needs 80 GB cards, and the reason for:

### LoRA: training 1 % of the weights

Low-Rank Adaptation freezes `W` and adds `ΔW = B·A` with `A: [r][in]`, `B: [out][r]`, rank
`r` = 8–64. The forward becomes `y = W·x + B·(A·x)` — one extra small matvec pair per
projection, which slots into `model.Linear.MatVec` — and only `A` and `B` get gradients and
optimizer state. With the base `W` still in Q4_K on the device this is **QLoRA**: fine-tune
the 3B on the 8 GB laptop card. Adapters are a few MB; the plan is to save them as GGUF
tensors (`blk.i.attn_q.lora_a` …) with the existing writer and merge or apply them at load.

### Data and the loop

A dataset of chat-formatted sequences, tokenized with `tokenizer/`, padded into batches
with attention masks and a loss mask over assistant tokens; a loop of forward → loss →
backward → optimizer step → log; a learning-rate schedule (warm-up, cosine decay); and an
evaluation on held-out data. The proof planned for Phase 5 is a loss curve on a small
dataset that matches a PyTorch run of the same LoRA configuration.

## How the phases build up to it

- **Phase 2's batched prefill** *is* the training forward pass: all positions of a sequence
  through the model at once with a causal mask. The `AttnPrefill` kernel and the batched
  GEMM path serve both.
- **Phase 4's tensor-core GEMM** is the backward pass's workhorse: `dW = dy ⊗ x` is a GEMM
  over the whole batch.
- **The CPU twins** give every backward kernel an oracle, and finite differences
  (`(loss(W+ε) − loss(W−ε)) / 2ε` on a handful of weights) give the whole gradient one.
- **The GGUF writer** already exists for saving results.

## Further reading

- Vaswani et al., *Attention Is All You Need* (2017) — the transformer.
- Touvron et al., *Llama 2* (2023) and the *Llama 3 Herd of Models* (2024) — the
  architecture and training recipe of the models used here.
- Hu et al., *LoRA* (2021); Dettmers et al., *QLoRA* (2023).
- Rafailov et al., *Direct Preference Optimization* (2023).
- Karpathy's *llm.c* and *nanoGPT* — training loops small enough to read in full.
