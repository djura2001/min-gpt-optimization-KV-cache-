# Figure 3 methodology alignment before the final Colab sweep

## Context

The implementation fixes from the previous pass all landed correctly — the `forward`/training API is restored, `vanilla=True` is the default again, KV memory uses `cache_len` per model, the α→1 limit is right, `g_per_round` handles partial rounds, `idx_cond` is used in the vanilla path, and the `max_new_tokens <= 0` guard is in. The central KV cache and greedy speculative decoding logic looks correct.

What remains is experimental methodology for Figure 3, not a bug in the decoding algorithm. Four items, all of which must land **before** the final GPU sweep — after the sweep, fixing any of them would require another GPU session.

Do not implement Phase B (stochastic acceptance) or int8 KV cache quantization; both remain out of scope.

---

## 1. `v_γ` must be measured on the target model, per target model

### What's wrong

The T_q scan currently builds its own model:

```python
tq_model = GPT.from_pretrained('gpt2', vanilla=False).to(DEVICE).eval()
tq_scan_df = measure_tq_scan(tq_model, prompt_len=64)
```

`v_γ` is defined as

**v_γ = t_target(T_q = γ+1) / t_target(T_q = 1)**

so it must be measured on the model that actually performs verification. In the main experiment that is `gpt2-medium`, not `gpt2`. The scan currently produces `v_γ` for `gpt2` and Figure 3 consumes it as if it were `gpt2-medium`'s.

This matters because `v_γ` measures how much cheaper a wide verification pass is relative to a narrow one, which depends on where the model sits relative to the hardware's ridge point. That position is not the same for two models of different size, and the direction of the difference is not predictable a priori — it has to be measured, not assumed.

### Also: it must be per target model, not global

There is a single global `tq_scan_df`, and `get_v_gamma()` reads from it. The `gpt2-large` branch already measures its own `c_ratio_large`, but there is no corresponding per-model `v_γ`.

This is currently **latent, not active** — the `gpt2-large` branch computes `c_ratio_large`, throughput and `speedup_vs_vanilla`, but does not call `get_v_gamma` or `predicted_speedup`, so no wrong number is being produced today. But `gpt2-large` is the configuration most likely to show speedup above 1.0, so it will almost certainly get the analytical curve once results come in. Making the scan target-specific now avoids needing a second GPU session then.

### Fix

Restructure so the scan is keyed by target model:

```python
tq_scans = {
    'gpt2-medium': measure_tq_scan(target_medium, prompt_len=representative_past_len),
    # added conditionally, same gating as the existing gpt2-large branch:
    'gpt2-large':  measure_tq_scan(target_large,  prompt_len=representative_past_len),
}
```

and have `get_v_gamma` take the relevant scan as an argument:

```python
get_v_gamma(tq_scans['gpt2-medium'], gamma)
```

Reuse the already-loaded target models rather than constructing fresh ones where possible, to avoid extra GPU memory pressure. Keep the existing assertion that the scan covers `T_q = γ+1` for every γ in `SPEC_GAMMAS`.

---

## 2. `representative_past_len` for the T_q scan

`measure_tq_scan(model, prompt_len=...)` first builds a cache of that length, then passes `T_q` new tokens through. So that parameter is the **past length**, and the total attended length during a verification pass is `T_k = past_len + T_q`. Naming it after `T_k` would misdescribe what the function does.

During a speculative run the cache grows from `SPEC_PROMPT_LEN` to roughly `SPEC_PROMPT_LEN + SPEC_MAX_NEW_TOKENS`, so the natural single representative point is the midpoint:

```python
# Representative past (cache) length, not T_k: during a verification pass the target
# attends over T_k = past_len + T_q. Rather than scanning the full 2-D (T_k, T_q)
# space, we measure v_gamma at a cache length roughly halfway through generation.
representative_past_len = SPEC_PROMPT_LEN + SPEC_MAX_NEW_TOKENS // 2
```

Use this as the `prompt_len` argument for every T_q scan. Keep the comment — the distinction goes into the thesis methodology chapter verbatim.

---

## 3. Match measurement conditions across `c`, `v_γ`, and measured speedup

`measure_cost_ratio(draft_model, target_model)` is called with its defaults (`prompt_len=32`, `max_new_tokens=20`), while the main sweep uses `SPEC_PROMPT_LEN` and `SPEC_MAX_NEW_TOKENS = 30`. The T_q scan used `prompt_len=64`.

All three quantities feeding Figure 3 — `c`, `v_γ`, and the measured speedup — should be measured under comparable context conditions, otherwise "measured vs. analytical prediction" is comparing across different experimental setups.

```python
c_ratio, draft_result, target_result = measure_cost_ratio(
    draft_model, target_model,
    prompt_len=SPEC_PROMPT_LEN,
    max_new_tokens=SPEC_MAX_NEW_TOKENS,
)
```

Apply the same to `c_ratio_large` in the `gpt2-large` branch.

---

## 4. Average `S(α_i)` over prompts instead of `S(mean α)`

### What's wrong

```python
mean_alpha = sub['alpha_conditional'].mean()
idealized_speedup = predicted_speedup(gamma, mean_alpha, ...)
```

`S(α)` is nonlinear, so `S(E[α]) ≠ E[S(α)]`. `S` is convex in α over the relevant range, so by Jensen's inequality `E[S(α)] ≥ S(E[α])`: the current curve **underestimates** the predicted speedup.

This matters more than the size of the error alone suggests, because it points in the opposite direction from the `v_γ` error in item 1. The two partially cancel, which is the worst case — it makes the current plot look more consistent than it is. Fix both in the same pass; fixing only one would produce an intermediate state that looks worse than the current one for the wrong reason.

### Fix

There are already 16 individual prompts with their own `alpha_conditional`. Compute the prediction per prompt and average the results:

```python
per_prompt_speedups = [
    predicted_speedup(gamma, alpha_i, denom_v)
    for alpha_i in sub['alpha_conditional']
]
theory_speedup = sum(per_prompt_speedups) / len(per_prompt_speedups)
```

Also report the spread across prompts (min/max or std), the same way other metrics report median and range — the variation across prompt categories is itself a result.

Keep `c_ratio` and `v_γ` as single measured constants inside this loop; only α varies per prompt.

---

## Order of work

1. Items 1 and 2 together (per-target `tq_scans` keyed by model name, `representative_past_len`).
2. Item 3 (matched conditions for `c`).
3. Item 4 (per-prompt theoretical curve) — must land in the same pass as item 1, per the cancellation note above.
4. Local CPU run with toy models to confirm the pipeline is intact end to end; real numbers still require the Colab GPU run.

After this pass the code should be frozen as the version that goes into the thesis, so flag anything you find that looks like it would need another change after the sweep.

---

## Resolution log (applied 2026-09-07)

All four items verified against the actual notebook (not just the prose) before fixing — including confirming item 1 as a real bug in the *previous* fix pass (the T_q scan I'd added was measuring `gpt2`, the draft's architecture, not `gpt2-medium`, the target that performs verification), and confirming item 4's convexity claim analytically (`S(α)` is a positive-weighted sum of `α^k` terms for `k=0..γ`, each convex on `[0,1]`, so the sum is convex and Jensen's inequality applies directly).

This required restructuring cell order, not just patching values in place: `PROMPT_CATEGORIES`, `SPEC_PROMPT_LEN` (tokenizer-resolved), `SPEC_MAX_NEW_TOKENS`, `SPEC_GAMMAS`, `SPEC_N_REPS`, and the `target_cached`/`draft_cached` model loads all moved from the real-sweep cell to the T_q-scan cell, since the T_q scan and cost-ratio diagnostics need them and previously ran *before* they were defined (working around this with hardcoded stand-ins is exactly what caused item 1). The real-sweep cell now just reuses these globals plus loads the two vanilla models it alone needs.

- `tq_scans` is now a dict keyed by target model name (`'gpt2-medium'`, with `'gpt2-large'` added conditionally in the bonus cell, reusing the shared `representative_past_len`); `get_v_gamma(tq_scan, gamma)` takes the scan explicitly instead of reading a bare global.
- `representative_past_len = SPEC_PROMPT_LEN + SPEC_MAX_NEW_TOKENS // 2`, used consistently for every T_q scan.
- `measure_cost_ratio` now takes `prompt_len`/`max_new_tokens` as required (not defaulted) arguments; both the main and `gpt2-large` cost-ratio calls pass `SPEC_PROMPT_LEN`/`SPEC_MAX_NEW_TOKENS` explicitly.
- Figure 3 now computes `predicted_speedup` per prompt's own `alpha_conditional` and averages the *results*, for both the idealized and v_γ-corrected curves, plotting the per-prompt min/max as error bars rather than a single point derived from the mean α.

Local validation: full `bench.ipynb` (47 cells) executes with zero errors as a CPU pipeline check. T_q scan now correctly measures `gpt2-medium` (median 0.055s at T_q=1, vs. the old and wrong `gpt2` measurement's 0.020s) and covers T_q=1..9. Cost ratio under matched `SPEC_PROMPT_LEN`/`SPEC_MAX_NEW_TOKENS` conditions came out to c=0.369 (previously 0.401 under unrelated default conditions — a different number for a legitimate reason, not noise). `mingpt/model.py` and the test suite were untouched this round (9/9 still passing, unaffected).

**All speculative-decoding numbers and Figure 3 curves collected before this fix are invalid** — same standing requirement as the previous two passes.
