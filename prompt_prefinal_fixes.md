# Pre-final-sweep fixes: API regression, measurement correctness, and the v_γ model

## Context

The draft-cache desync and the `alpha_conditional` separation from the previous pass are both correctly implemented — full-accept synchronization, the `rollback_kv_cache` guard, and `tested` vs. `proposed` all look right. This pass fixes a different set of issues found in a subsequent review, all of which must land **before** the final Colab GPU sweep.

Two of these (items 1 and 6) are regressions against the original minGPT repository, not just measurement bugs. The complete code goes into the thesis appendix, so a repo where standard minGPT training no longer runs is a real problem independent of the benchmarks.

Do not implement Phase B (stochastic acceptance) or int8 KV cache quantization — both remain out of scope.

---

## P0-1. Restore the original `GPT.forward` API and the training path

### What's wrong

Current signature:

```python
def forward(self, idx, kv_cached, targets=None):
```

`kv_cached` is a required second positional argument. This breaks two things:

- The HuggingFace comparison path still calls `model(x1)` → `TypeError`.
- `trainer.py` calls `model(x, y)`, so the targets tensor `y` is now bound to `kv_cached`. **Standard minGPT training is broken.**

Compounding this: `GPT(...)` and `GPT.from_pretrained(...)` currently default to `vanilla=False`, so they construct cached attention, and `CausalSelfAttention.forward` appends to the cache on every call. Even with a compatible signature restored, training would accumulate cache across minibatches.

### Fix

Restore a backward-compatible signature with `kv_cached` as a trailing keyword:

```python
def forward(self, idx, targets=None, kv_cached=False):
```

Convert every internal inference call site to keyword form: `self(idx_cond, kv_cached=False)`, `self(idx_next, kv_cached=True)`, and the same in `generate_speculative` for both target and draft.

Restore `vanilla=True` as the default for both `GPT.__init__` and `GPT.from_pretrained`. The benchmark already selects explicitly via `vanilla=not cached`, so nothing in `bench.ipynb` depends on the old default.

### Critical follow-up

Because the `vanilla` default flips, **audit every construction site** in `bench.ipynb` and `tests/test_speculative.py` and make `vanilla=` explicit at each one. If a test silently falls back to `vanilla=True` on both sides, the identity tests degrade to comparing vanilla against vanilla, which passes trivially and verifies nothing. This is the main risk in this whole change — check it deliberately, don't assume.

`README` and `demo.ipynb` use `GPT(model_config)` with no `vanilla` argument; after this fix they work again unchanged. Verify `trainer.py` runs at least a few iterations.

---

## P0-2. Analytical KV memory is one token too large (Figure 2)

### What's wrong

`bench.ipynb` currently does:

```python
kv_measured = measured_kv_cache_bytes(model)
seq_len = out.size(1)
kv_analytical = analytical_kv_cache_bytes(..., seq_len)
```

But cached `generate()` maintains the invariant `T_cache = T_output − 1`: the last generated token has not yet been passed through a forward, so it is not in the cache. The comparison is therefore between `2BLd(T−1)b` measured and `2BLdTb` analytical, a systematic discrepancy of exactly `2BLdb` bytes.

### Fix

Derive the analytical length from the cache itself, and take the dtype from the cache tensor rather than assuming:

```python
cache_len = model.transformer.h[0].attn.k_cache.size(2)
kv_analytical = analytical_kv_cache_bytes(
    out.size(0),
    model.num_layers,
    model.transformer.wte.embedding_dim,
    cache_len,
    dtype=model.transformer.h[0].attn.k_cache.dtype,
)
```

Apply the same fix in `run_once_speculative`, computed **separately for target and draft** — their final cache states don't necessarily carry the same lag, especially after the full-accept synchronization path.

---

## P0-3. Wrong limit for α = 1 in the theoretical speedup

### What's wrong

```python
if abs(1 - mean_alpha) < 1e-9:
    theoretical_speedup = gamma + 1
```

That is the limit of the numerator alone, not of the full expression. For

S(γ) = (1 − α^(γ+1)) / ((1 − α)(1 + γc))

the limit as α → 1 is:

**S(γ) = (γ + 1) / (1 + γc)**

### Fix

```python
if abs(1 - mean_alpha) < 1e-9:
    theoretical_speedup = (gamma + 1) / (gamma * c_ratio + 1)
```

This branch does fire in the existing data — the `code_0` prompt produced α = 1.0 at every γ — so it is not hypothetical.

---

## P0-4. Replace the idealized formula with the measured v_γ model

This is the highest-value item in this pass. It is not just a renaming.

### What's wrong

The notebook calls the analytical curve a "theoretical bound" and warns when measured speedup exceeds it by more than 15%. But the formula

S(γ) = E[N_out] / (1 + γc)

**assumes** that verifying γ+1 tokens with the target costs approximately the same as one target step. That assumption is exactly what the existing T_q scan measures — and it is not guaranteed to hold.

### Fix

Generalize the model using the already-measured quantity:

**S(γ) = E[N_out] / (γc + v_γ)**,  where  **v_γ = t_target(T_q = γ+1) / t_target(T_q = 1)**

Compute `v_γ` per γ from the existing T_q scan results (extend the scan's T_q range if it doesn't currently cover γ+1 for every γ in the sweep). If the GPU shows v_γ ≈ 1, this reduces to the original formula, and that agreement is itself a result worth reporting.

Plot both curves in Figure 3: the idealized prediction (v_γ = 1) and the v_γ-corrected prediction, alongside the measured speedup. The gap between the two curves is a direct measurement of how well the memory-bound assumption holds on this hardware.

Rename the label from "theoretical bound" to "analytical prediction" / "idealized prediction" throughout the notebook, and soften the warning: exceeding an idealized prediction is informative, not necessarily an error.

**Why this matters for the thesis:** measured speedup is currently below 1.0 at every γ, with c = 0.50. Quantifying v_γ gives a second independent, measured explanation for that outcome rather than leaving it as an unexplained gap between theory and measurement.

---

## P1-5. E[A] validation ignores partial final rounds

### What's wrong

The last round runs with `g = min(gamma, remaining_budget)`, so it is frequently shorter than γ. With `SPEC_MAX_NEW_TOKENS = 30` and γ ∈ {1,2,3,4,5,6,8}, partial final rounds are common, not a rare edge case. But the analytical comparison computes the full-γ expectation:

```python
sum(mean_alpha_cond ** k for k in range(1, gamma + 1))
```

### Fix

Record `g_per_round` (the actual block length used in each round) in the stats dict, and compute the analytical expectation per-round against the true block length:

E[A_r] = Σ_{k=1}^{g_r} α^k

then average across rounds. This is more robust than tuning `SPEC_MAX_NEW_TOKENS` to a value divisible by every γ, because it doesn't depend on the constant staying that value.

---

## P1-6. Vanilla `generate()` computes `idx_cond` and then ignores it

### What's wrong

In the decoding loop (`model.py`, around line 413):

```python
idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
...
else:
    logits, _ = self(idx, False)   # <-- idx, not idx_cond
```

The crop is computed and discarded. Once a vanilla sequence exceeds `block_size`, `forward` asserts on `t > block_size`. Current measurements use short enough contexts that this doesn't affect results, but it's a live bug in the appendix code.

### Fix

```python
else:
    logits, _ = self(idx_cond, kv_cached=False)
```

---

## P2-7. `max_new_tokens=0` still generates one token

### What's wrong

Cached `generate()` performs the prefill unconditionally, appends `idx_next`, and only then does `max_new_tokens = max_new_tokens - 1`. `generate_speculative` has the same structure. So `max_new_tokens=0` returns one extra token. The overshoot test only covers 1 through 5.

### Fix

Early return at the top of both functions:

```python
if max_new_tokens <= 0:
    return idx
```

For `generate_speculative` with `return_stats=True`, return an appropriately-shaped empty stats dict (zero counters, empty lists, alpha as `float('nan')` rather than a division by zero). Extend the overshoot test to include `max_new_tokens=0`.

---

## Sweep configuration changes for the final run

Beyond the fixes above, the sweep itself needs one change before it's worth spending GPU time on.

The measured cost ratio is **c = 0.50** for gpt2 (draft) / gpt2-medium (target). At that value the analytical ceiling is `(γ+1)/(1+γc)`, which is 1.33 at γ=1 and falls below 1.0 from γ=2 onward at realistic α. No correctness fix changes this: after all the above, measured and predicted speedup should finally agree, but both will stay below 1.0 on this model pair.

To also capture a regime where the method does help, add to the same sweep run (a second Colab session is expensive in the remaining time budget):

- **`gpt2-large` as a second target**, with `gpt2` as draft, gated on available GPU memory. The larger size ratio should push c well below 0.50. If memory allows, `gpt2-xl` as a third point is better still.
- **Longer prompts for the speculative runs.** The current `SPEC_PROMPT_LEN` is short, which is the worst case for c — the target's per-call fixed overhead dominates most when context is short. Sweep at least two or three prompt lengths (e.g. short / medium / long) so the c-versus-context relationship is visible.
- **Replace the `code_0` prompt.** It produced α = 1.0 at every γ (26/26 accepted at γ=8), which almost certainly means it entered a degenerate repetition loop where both models trivially agree. It is responsible for the large error bars on the `code` category in Figure 4. Print the generated continuation for each prompt once so this class of degeneracy is visible rather than silent.

---

## Order of work

1. P0-1 (API restore + `vanilla` default flip), then immediately audit all `vanilla=` call sites and run the full test suite plus a few `trainer.py` iterations.
2. P0-2, P0-3, P0-4 (measurement and formula correctness).
3. P1-5, P1-6, P2-7.
4. Sweep configuration changes.
5. Local CPU run with toy models to confirm the pipeline is intact end to end. Real numbers still require a Colab GPU run.

All previously collected speculative-decoding numbers remain invalid and must be re-measured after these changes.

---

## Resolution log (applied 2026-09-07)

Every claim verified against the actual code (and, where feasible, by running it) before fixing, not taken on faith:

- **P0-1**: reproduced both breakages directly. `tests/test_huggingface_import.py` failed with `TypeError: GPT.forward() missing 1 required positional argument: 'kv_cached'`. A toy `trainer.py` run failed with `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` (the target tensor `y` was binding positionally to `kv_cached`, then getting evaluated as a condition). Both now pass/run after the fix — confirmed by re-running the HF test (`ok`) and a real 5-iteration toy training loop (completed, reported a loss value).
- **P0-2**: reproduced numerically — after `generate()`, `out.size(1)=18` but the actual cache length was `17`, confirming the systematic one-token overstatement in both `run_once` and `run_once_speculative` (the latter previously didn't even split target vs. draft cache size at all; it used only the target's architecture for both).
- **P0-3**: confirmed by reading the formula — the `alpha≈1` branch returned `gamma+1`, dropping the `/(gamma*c_ratio+1)` denominator entirely.
- **P1-6**: confirmed at the exact line — `self(idx, False)` uses `idx`, not the `idx_cond` computed two lines above.
- **P0-4**: confirmed the T_q scan's `tq_values=(1,5,9)` covers only 2 of the 7 `gamma+1` values actually needed by `SPEC_GAMMAS=[1,2,3,4,5,6,8]`.
- **P1-5**: confirmed partial final rounds are the common case, not an edge case, given `SPEC_MAX_NEW_TOKENS=30` and `SPEC_GAMMAS` up to 8.

Fixes applied:

- `mingpt/model.py`: `forward(self, idx, targets=None, kv_cached=False)` restored; every internal call site (`generate`, `generate_speculative`, both target and draft) converted to keyword form (`kv_cached=...`); `GPT.__init__` and `GPT.from_pretrained` default back to `vanilla=True`; `generate()`'s vanilla decode branch now uses `idx_cond` (P1-6); both `generate()` and `generate_speculative()` early-return on `max_new_tokens <= 0`, with `generate_speculative` returning a well-shaped empty stats dict (P2-7); `generate_speculative` now also tracks `g_per_round` alongside `accepted_per_round` (P1-5 plumbing).
- Audited every `GPT(...)`/`GPT.from_pretrained(...)` call site in `bench.ipynb` and `tests/test_speculative.py`: all were already explicit about `vanilla=`, so the default flip changes nothing there — confirmed by re-running the full test suite (9/9 passing) after the change.
- `bench.ipynb`: `run_once`/`run_once_speculative`/T_q scan converted to keyword calls; KV analytical calculation in both now reads the real cache length and dtype instead of `out.size(1)` (P0-2), with `run_once_speculative` computing it separately for target and draft (`kv_cache_analytical_bytes_target`/`_draft`); T_q scan extended to `tq_values=(1..9)` so `v_gamma` is computable for every swept γ (P0-4); Figure 3 now plots measured / idealized (`v_gamma=1`) / v_gamma-corrected curves, with the α→1 limit fixed to include the `(gamma*c+v)` denominator (P0-3); the E[A] validation cell now compares each row's measured E[A] against an analytical value computed from that row's own actual `g_per_round` sequence, not a blanket `range(1, gamma+1)` (P1-5).
- Not implemented: the "sweep configuration changes" section (`gpt2-large` target, longer prompts, replacing the degenerate `code_0` prompt) — flagged to the user as a scope/GPU-time decision rather than a bug fix, since it expands the experiment rather than correcting it.

Local validation: full test suite 9/9 passing (`tests/test_speculative.py` + `tests/test_huggingface_import.py`); full `bench.ipynb` (47 cells) executes with zero errors as a CPU pipeline check, T_q scan now reports all 9 values.

**All speculative-decoding numbers collected before this fix are invalid and must be re-measured on Colab GPU** (compounding the same requirement already stated after the previous fix pass).
