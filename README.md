# LLM determinism experiment

Same prompt, temperature 0 – different answers? This repository contains the scripts and raw data
of a small experiment that checks, cause by cause, where non-deterministic output of a locally
served LLM comes from.

## Setup

| | |
|---|---|
| Hardware | AMD Strix Halo (Radeon 8060S, gfx1151), 128 GB RAM, Ubuntu 26.04 |
| Software | [llama.cpp](https://github.com/ggml-org/llama.cpp) commit `749db7f`, Vulkan backend (Mesa 26.0.8), flash attention on |
| Model | Qwen3-8B, `Qwen3-8B-Q4_K_M.gguf`, thinking disabled (empty `<think>` block in the prompt) |
| Prompt | "Write a short story about a lighthouse keeper who finds a strange message in a bottle. Be creative." |
| Output | 200 tokens per request |
| Date | 2026-11-18 to 2026-11-21 (main series on 2026-11-18) |

"Same answer" means byte-identical generated text (compared via SHA-256).
Each test uses a dedicated `llama-server` instance on `127.0.0.1` that serves no other users,
except in s8, where concurrent requests are added on purpose.

## Results

| Series | Test | Runs | Distinct outputs | Most frequent |
|---|---|---|---|---|
| s1 | temperature 0.7, no seed | 30 | 30 | 1× |
| s2 | temperature 0.7, seed 42 | 30 | 7 | 24× |
| s3 | temperature 0, sequential, 1 slot, no prompt cache | 30 | 3 | 28× |
| s4 | temperature 0, prompt cache on | 30 | 4 | 15× |
| s5 | prompt processing only (first-token logprobs) | 30 | 1 | 30× |
| s6 | temperature 0, fresh server per request | 10 | 1 | 10× |
| s7 | temperature 0.7 + seed 42, fresh server per request | 5 | 1 | 5× |
| s8 | temperature 0, 3 concurrent requests from other "users" (4 slots) | 30 | 6 | 15× |
| s9 | temperature 0, 4 slots, sequential (control for s8) | 30 | 6 | 25× |
| v1 | temperature 0, one `llama-completion` process per request | 15 | 1 | 15× |
| v2 | temperature 0, running server, slot erased before each request | 15 | 5 | 7× |
| seq | temperature 0, 16 requests after a fresh start, 3 fresh starts each, without / with prompt cache | 3 × 16 each | see below | — |

s1–s5 ran one after another on the same server instance (one start), so each series inherits the
history of the previous ones. s8 and s9 each ran on a freshly started 4-slot server. Given the
history dependence below, this matters: s4 starts differently from the fresh-start cache sequence,
but ends in the same alternation of the same two answers.

**The server as a whole is deterministic; the answer depends on its history** (`sequence.sh`).
The same 16 requests after a fresh start produce the same sequence of answers on every repetition
(letters = distinct outputs):

```
no cache (prompt fully recomputed every time, prompt_n = 33):  A A B C D C E E E D D D D D D D   (3/3 identical)
prompt cache on (prompt_n = 1 from the 2nd request on):          A A F A A B E E E F E F E F E F   (3/3 identical)
```

Observations:

- At temperature 0, almost all divergences start at the same token (index 102 or 112): a point where two
  continuations are nearly equally likely.
- Run order matters. Without concurrency and without prompt cache (s3, s9), only the first few runs of a
  series deviate; runs 11–30 are 20/20 identical. With concurrent requests (s8), runs 11–30 still show
  3 variants. With prompt cache (s4), two answers alternate strictly from run 8 on.
- s8 does not separate batching from history: the concurrent requests change the batch, but they
  are also part of the server's history. The clean batching test is the vLLM logprob comparison
  below (no history dependence there, yet concurrent requests shift the numbers). One data point
  without prior history (n = 1): the very first request of s8, on a fresh 4-slot server, gave
  `d699…`, while a fresh 4-slot server without concurrent requests gave `bf3c…` in 8/8 cases
  (slots_first_4, s9, slots_seq4).
- The configuration changes the answer, already for the very first request on a fresh server
  (`slots_check.sh`): 1 slot → `025aef0fbb` in 15/15 fresh starts (s6 plus 5 more); 2 slots and 4 slots →
  `bf3c…` in 5/5 fresh starts each. The 16-request sequence on a fresh 4-slot server (twice) equals the
  first 16 runs of s9. Do not compare the *settled* answers of s3 (1 slot, 216f…) and s9 (4 slots,
  6aad…): a fresh 1-slot server settles on yet another text (eac5…, `seq_nocache_*`), so the settled
  answer depends on the history as well.
- Prompt processing is bit-identical across runs (s5). The kernels themselves are run-to-run deterministic
  on this setup.
- Switching flash attention off changes the numbers, but they stay run-to-run identical: with the short
  probe prompt of `probe.py`, the favorite " a" moves from 55.83 % to 55.07 % on a fresh server
  (`results/near_tie_fa_probe.txt`).
- A fresh process gives identical output (s6, s7, v1); a long-running server does not, even with one slot,
  no cache and no concurrent requests (s3). Erasing the slot through the API does not help (v2): v2 even
  repeats the first 15 answers of the fresh-start sequence (`seq_nocache_*`) exactly.
- The prompt cache is not the cause by itself: with the cache on, the server recomputes exactly one prompt
  token on every request from the 2nd on, yet the answers still change (`cache_probe.py`).
- **Not proven:** *which* state carries the history. A plausible candidate from reading the code is the
  KV cache layout (attention length `n_kv`, padded to multiples of 256; the Vulkan flash-attention kernel
  derives its split-k chunk boundaries from it). This was not measured.
- **CPU only, no history dependence** (`cpu_check.sh`, 1 slot, two fresh starts × 12 requests,
  2026-11-21): 24/24 identical (`282c…`). Important detail: `-ngl 0` alone does *not* keep the GPU
  out. llama.cpp still offloads ops with a batch of ≥ 32 tokens to the GPU (`GGML_OP_OFFLOAD_MIN_BATCH`,
  default 32), and the prompt has 33 tokens. A first run with only `-ngl 0` (2026-11-20, kept as
  `results/cpu_offload_seq_*.jsonl`) did show the effect (`025a…` first, then `12f7…`, identical on
  both starts). With `--device none --no-op-offload` it is gone. So in this setup the history
  dependence needs the GPU (Vulkan) to be involved, which fits the candidate above but does not
  prove it (24 runs).
- **Pitfall found on the way:** `llama-completion -f` strips one trailing newline from the prompt file.
  For this prompt that turns the last token `"\n\n"` (271) into `"\n"` (198) and produces a completely
  different story from the very first token. `variants.sh` compensates for it now; the recorded
  `results/v1_cli.txt` was produced *before* the fix (consistent within itself, but a different prompt
  than the server runs). With the fix, the CLI produces the same text as a fresh server (checked 3×,
  2026-11-21). Its stdout ends with an extra blank line, though, so a re-run of `variants.sh` records a
  different hash than the server runs (`eef194c33b` instead of `025aef0fbb`).

## Second model: gpt-oss-120b

The core tests repeated with a very different model (`second_model.sh`): gpt-oss-120b (MoE, 117B
parameters, MXFP4 GGUF), same hardware and llama.cpp build, same task in the model's own chat format
(`prompts/gpt-oss.txt`, reasoning effort "low", date fixed), temperature 0, 1 slot, no prompt cache.

| Series | Test | Runs | Distinct outputs | Most frequent |
|---|---|---|---|---|
| g1 | long-running server, sequential | 30 | 26 | 3× |
| g2 | fresh server per request | 10 | 1 | 10× |
| g3 | 16 requests after a fresh start, 3 fresh starts | 3 × 16 | 16 per start | sequence 3/3 identical |

The g3 sequence is also identical to runs 1–16 of g1. Most divergences start at token index 28
("The lighthouse at Grayhaven …" vs. "… on Gull’s Point …"). Same pattern as with Qwen3-8B, but much
stronger; why (size, quantization, architecture) was not investigated.

## Second software: vLLM with and without batch-invariant mode

Separate machine: NVIDIA RTX 4070 (12 GB, compute capability 8.9), driver 595.99.02, vLLM 0.34.0,
torch 2.14.0+cu130, `Qwen/Qwen3-8B-AWQ` (revision 4da05a8e), fp16, `--max-model-len 8192 --seed 0`,
prefix caching at vLLM's default, `/v1/completions`, temperature 0, same prompt as above.
`VLLM_BATCH_INVARIANT=1` switches the batch-invariant mode on.

Text comparison (`vllm_check.sh`): **216/216 identical**, in both modes, sequential (30), with
3 concurrent strangers (30) and 3 × 16 after fresh starts. Both modes even produce the same text.

That alone proves little: the two modes use different kernels, yet give the same text. So
`vllm_logprobs.sh` compares the logprob of every generated token at full precision:

| | default mode | batch-invariant mode |
|---|---|---|
| sequential, 10 runs | 1 logprob set (bit-identical) | 1 |
| 3 concurrent strangers, 10 runs | **4 different sets**, ~180 of 200 tokens shifted, max \|Δlogprob\| 0.0166 | **1** |
| another fresh start, 5 runs | identical to the first start | identical |
| default vs. batch-invariant | max \|Δlogprob\| 0.0155, same text | |
| time per request | ≈ 2.5 s | ≈ 30 s (≈ 12×) |

Observations:

- Concurrent requests change the numbers in default mode, but the favorite never flipped. Without
  concurrency the closest call is token 59, "," 34.90 % vs. " had" 34.35 %. With concurrent requests,
  individual runs show **exact ties**: token 59 "," vs. " had" both 34.612 %, token 113 "people" vs.
  "folk" both 50.0 % (`vllmlp_default_noise`). A tie is resolved the same way in every run, so the text
  stays equal, but this is closer than "no token was close enough". The text comparison simply was not
  sensitive enough for this model/prompt.
- Batch-invariant mode removes the effect completely.
- vLLM showed **no history dependence** for sequential requests, unlike llama.cpp above. Software, GPU
  vendor and quantization (AWQ vs. Q4_K_M) differ at the same time, so this does not say which of them
  matters. (Partly resolved by the third model below: neither llama.cpp as a whole nor quantization.)
- In batch-invariant mode, token 113 is an exact tie ("people" and "folk", both logprob −0.693155,
  i.e. 50.0 %) in every run; it is still resolved the same way every time.
- The 12× cost is higher than the ≈2× reported by Thinking Machines, but not comparable: they measured
  1,000 requests together (26 s → 55 s), these are single requests. Likely reason for our factor (not
  measured): for AWQ models vLLM disables its Marlin kernels in this mode ("not batch invariant")
  and falls back to dequantize + matmul. Timings are rough, from log timestamps
  (`results/vllm_request_log.txt`). With the unquantized third model below, the factor is 2.0.

## Third model: Qwen3-1.7B, unquantized, on both engines

Repeats the core tests with a small model, many runs and the **same weights everywhere**
(`small_model.sh`, 2026-11-22, 2,264 runs): `Qwen/Qwen3-1.7B`, revision `70d244cc`, BF16
safetensors (identical SHA-256 on both machines), computed in FP16 by every engine:

- llama.cpp: F16 GGUF made with llama.cpp's `convert_hf_to_gguf.py --outtype f16` (one-dimensional
  tensors such as norms stay F32); the same GGUF file on both machines. Same flags as above
  (`-ngl 99 -fa on -c 16384`, no prompt cache).
  - **Vulkan**: the Strix Halo machine and build from the main series.
  - **CUDA**: the RTX 4070 machine from the vLLM test, same commit `749db7f`, CUDA 12.4 (Ubuntu
    `nvidia-cuda-toolkit`), `GGML_CUDA=ON`, architecture 89.
- vLLM 0.34.0 as above, `--dtype float16`, FlashInfer disabled as in the 8B runs.

Same prompt and chat format as Qwen3-8B, temperature 0, 200 tokens. Every request records the logprob of
every generated token and its wall time (`measure.py`, `LOGPROBS=1`; for llama.cpp this reads the logits
after the token was chosen and does not change the computation, checked by a control series without
logprobs). Cells: distinct texts / distinct logprob sets.

| Series | llama.cpp Vulkan (Strix Halo) | llama.cpp CUDA (RTX 4070) | vLLM default | vLLM batch-invariant |
|---|---|---|---|---|
| run: 1 server, 300 sequential | 1 / 5 (requests 1–4 differ, 5–300 identical) | 1 / 1 | 1 / 1 | 1 / 1 |
| seq: 32 after a fresh start, 3 starts | same sequence on every start | identical | identical | identical |
| fresh: new server per request, 20 | 1 / 1 | 1 / 1 | – | – |
| slots4: 4-slot server, 100 sequential | 1 / 5 (requests 1–4 again), **other text than 1 slot** | 1 / 1, same as 1 slot | – | – |
| noise: 100 with 3 concurrent strangers | **2 / 18** | 1 / 4 | 1 / 2 | 1 / 1, same as sequential |
| median time per request, run / noise | 4.54 / 10.44 s | 1.69 / 2.18 s | 1.62 / 1.70 s | 3.25 / 3.24 s |

Observations:

- **History dependence follows the backend/GPU, not llama.cpp as a whole and not the quantization.**
  The same llama.cpp commit with the same F16 weights shows it on Vulkan (Strix Halo), not on CUDA
  (RTX 4070): there, 416 single-slot requests (run, seq, fresh) and 100 four-slot requests are
  bit-identical, logprobs included. llama.cpp's Vulkan backend, the Mesa driver and the AMD GPU change
  together; which of them matters is open (a Vulkan build on the RTX 4070 would separate the backend
  from the GPU, not done). Only one model was tested on CUDA (this one, not Qwen3-8B Q4_K_M).
- On Vulkan the history is a **warm-up**: after every start, requests 1–4 differ in their logprobs
  (in 184, 143, 125 and 80 of 200 tokens from request 5), from request 5 on they stay bit-identical
  for the remaining 296 requests. The text never flips with this model and prompt; the closest call
  is token 157, " strange" 32.59 % vs. " script" 32.37 %. With Qwen3-8B (Q4_K_M) the *texts* still
  changed up to request 10 after a start (seq above); logprobs were not recorded there, so the two
  warm-ups cannot be compared directly.
- Request 1 on a fresh Vulkan server is the outlier, not the settled state: first token " The" 73.24 %
  (logprob −0.311) vs. 77.22 % (−0.259) from request 5 on; CUDA gives −0.264, vLLM −0.265.
- **Number of slots** changes the answer on Vulkan (as with Qwen3-8B), not on CUDA. The 1-slot and
  4-slot texts split at token 155: " of" 45.99 % vs. "." 45.47 % with 1 slot, "." 45.709 % vs.
  " of" 45.701 % with 4 slots.
- **Concurrent requests** shift the numbers in every setup except batch-invariant mode. CUDA: all
  100 runs differ from the sequential run in 181–185 of 200 tokens (max |Δlogprob| 0.023). vLLM
  default: all 100 differ in 175–176 tokens, max 0.0165, the same size as with Qwen3-8B-AWQ (0.0166).
  Only on Vulkan did the text change: with strangers it splits off at token 73 (" lost" 26.57 % vs.
  " that" 26.52 %). After the warm-up (runs 1–4), runs 5–79 share one logprob set; runs 80–100
  add 13 more sets and 3 times the text of the 4-slot server without strangers. Why from run 80 on was not
  investigated; the machine had no other load (its router service logged no requests).
- **Same weights, different engines, different texts**: 3 distinct texts among the sequential
  answers of the 4 setups.
  llama.cpp CUDA and vLLM batch-invariant produce the same 200 tokens (logprobs differ by up to
  0.027). vLLM default leaves them at token 155, where it has an **exact tie** ("." and " of", both
  logprob −0.7819033); Vulkan leaves them at token 157 (" strange" instead of " script"). Between
  setups, logprobs on the shared tokens differ by 0.017 to 0.075.
- **Cost of batch-invariant mode: 2.0×** (3.25 s vs. 1.62 s per request) for the unquantized model,
  against ≈ 12× for Qwen3-8B-AWQ. This supports the Marlin explanation above, but does not prove it
  (the model size differs as well). Timings here are wall times per request from `measure.py`, not
  log timestamps.
- Controls: 10 requests without logprobs after a fresh start give the same texts as with logprobs
  (all 4 setups); the first 32 requests of `run` equal the `seq` sequence.
- **Pitfall on the way:** installing the CUDA toolkit *and* ninja on the vLLM machine (for the
  llama.cpp build) made vLLM enable FlashInfer, which JIT-compiles its sampling kernels at startup;
  this failed with nvcc 12.4. ninja was removed again (llama.cpp is built with the Makefile
  generator), and vLLM then produced output bit-identical to before. A first default-mode run of 300
  requests, started before the install, is bit-identical to the recorded one (not in `results/`).

## Files

| File | Purpose |
|---|---|
| `run_all.sh` | Series s1–s9 |
| `sequence.sh` | Same request sequence after fresh starts (results/seq_*.jsonl) |
| `vllm_check.sh`, `vllm_logprobs.sh` | vLLM default vs. batch-invariant (results/vllm_*.jsonl, results/vllmlp_*.jsonl); expects a vLLM start script, see `SERVE` |
| `second_model.sh` | g1–g3 with gpt-oss-120b (results/g*.jsonl); prompt in `prompts/gpt-oss.txt` |
| `variants.sh` | Series v1, v2 |
| `slots_check.sh` | First answer of a fresh server with 1 / 2 / 4 slots; 16-request sequence on 4 slots (results/slots_*.jsonl) |
| `cpu_check.sh` | Same request sequence on the CPU backend, GPU fully disabled (results/cpu_seq_*.jsonl); `results/cpu_offload_seq_*.jsonl` is the earlier run with `-ngl 0` only, where prompt processing was still offloaded to the GPU |
| `small_model.sh` | Third model (Qwen3-1.7B, FP16): `small_model.sh llama` (with `LABEL=vulkan` or `cuda`) and `small_model.sh vllm`; results/small_*.jsonl |
| `small_model_summary.py` | Summary of results/small_*.jsonl per setup and across setups; its output is `results/small_model_summary.txt` |
| `measure.py` | One measurement series against the test server (`/completion` endpoint); `PROMPT_FILE` overrides the built-in prompt; `API=openai` for vLLM; `LOGPROBS=1` records the logprob of every token (both APIs); every gen run records `secs` (wall time) |
| `analyze.py` | Summary of `results/*.jsonl` |
| `near_tie.py` | Probabilities of the candidates at a divergence point (e.g. `near_tie.py results/s6_t0_fresh.jsonl 102`: " had" 42.7 % vs. " kept" 41.9 %; `near_tie.py results/s4_t0_cache.jsonl 112`: " coast" 18.2 % vs. " ships" 17.7 %; both as the first request on a freshly started server, measured via prompt processing; the values shift with the server's history) |
| `cache_probe.py` | Prompt tokens actually recomputed per request with prompt cache on |
| `results/` | Raw data: one JSON line per run (hash, token IDs, text; `prompt_n` in seq_*) |
| `results/near_tie_fa_probe.txt` | Output of `near_tie.py` (token 102 and 112) and of `probe.py` with flash attention on/off, each on a fresh server (re-measured 2026-11-21) |
| `results/vllm_request_log.txt` | vLLM request timestamps (server log), source of the rough per-request times |
| `det_test.py`, `probe.py`, `gen_probe.py` | Earlier exploratory tests (chat endpoint, flash attention on/off, raw generation) |
| `publications/` | Published texts, one file per medium (`linkedin_article.md`, …) plus generated HTML |
| `build_html.sh` | Regenerates `publications/*.html` from the Markdown files (pandoc) |

## Publications

| Date | Medium | Version | Tag | Link |
|---|---|---|---|---|
| – | – | – | – | not published yet |

## Reproduce

Requirements: Linux with systemd user units, a llama.cpp build with `llama-server` and
`llama-completion`, Python 3 (standard library only), `curl`.

```bash
export MODEL=/path/to/Qwen3-8B-Q4_K_M.gguf      # default: ~/models/Qwen3-8B-Q4_K_M.gguf
export LLAMA_BIN=/path/to/llama.cpp/build/bin   # default: ~/llama.cpp/build/bin
export PORT=8091                                # default: 8091

./run_all.sh      # about 17 minutes on the hardware above
./variants.sh     # about 3 minutes
./sequence.sh     # about 10 minutes
./slots_check.sh  # roughly 20 minutes (mostly server starts)
./cpu_check.sh    # about 5 minutes
MODEL=/path/to/gpt-oss-120b-MXFP4.gguf ./second_model.sh   # about 9 minutes; the model file alone is 63 GB
python3 analyze.py results
```

For the vLLM comparison, `SERVE` must point to a script that starts `vllm serve` in the foreground
on 127.0.0.1 with the port as its first argument and the served model name `Qwen3-8B-AWQ`
(`VLLM_BATCH_INVARIANT` is passed in through the environment):

```bash
SERVE=/path/to/serve.sh ./vllm_check.sh      # about 65 minutes on an RTX 4070 (mostly batch-invariant mode)
SERVE=/path/to/serve.sh ./vllm_logprobs.sh   # about 17 minutes
```

Third model: download `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, convert it
with llama.cpp's own converter (`python convert_hf_to_gguf.py <model dir> --outtype f16`, requirements in
`requirements/requirements-convert_hf_to_gguf.txt`), and serve the safetensors with vLLM as `Qwen3-1.7B`
with `--dtype float16` (otherwise like `SERVE` above):

```bash
MODEL=/path/to/Qwen3-1.7B-F16.gguf LABEL=vulkan ./small_model.sh llama   # about 1 hour on the Strix Halo machine
MODEL=/path/to/Qwen3-1.7B-F16.gguf LABEL=cuda ./small_model.sh llama     # about 20 minutes on an RTX 4070
SERVE=/path/to/serve-qwen3-1.7b.sh ./small_model.sh vllm                # about 45 minutes on an RTX 4070
python3 small_model_summary.py results
```

On a machine that runs vLLM, do not put `ninja` on PATH next to a CUDA toolkit (see the pitfall above).

The exact numbers depend on hardware, driver, llama.cpp version and model file. Expect different hashes
on a different setup; the pattern (fresh process vs. long-running server, effect of concurrency) is what
the experiment is about.

## Related work

- Horace He, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), Thinking Machines Lab, 2025
- Yuan et al., [Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/abs/2506.09501), NeurIPS 2025, arXiv 2506.09501
- Shanmugavelu et al., [Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications](https://arxiv.org/abs/2408.05148), SC24 Workshops, 2024
- Ingonyama, [Solving Reproducibility Challenges in Deep Learning and LLMs: Our Journey](https://www.ingonyama.com/post/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey), 2024
- vLLM, [Batch Invariance](https://docs.vllm.ai/en/latest/features/batch_invariance.html); SGLang, [Deterministic Inference](https://docs.sglang.io/advanced_features/deterministic_inference.html)

Scripts written together with Claude (Anthropic).
