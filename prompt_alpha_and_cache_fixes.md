# Fix draft KV cache invariant + separate the three acceptance metrics

## Context

Two problems were found in `generate_speculative` (Phase A, greedy) in `mingpt/model.py`. Both must be fixed **before** the final Colab GPU measurement run, because both silently corrupt the acceptance-rate numbers that feed Figure 3 and Figure 4 of the thesis, while leaving the existing identity tests passing.

`tests/test_speculative.py` currently passes in full. Do not weaken or delete any existing test. Add new tests as described below.

---

## Problem 1: draft KV cache is one token short after a fully-accepted block

### What's wrong

In the drafting loop:

```python
cur = idx[:, -1:]
for _ in range(g):
    dlogits, _ = draft_model(cur, True)
    dnext = self._apply_temp_topk(dlogits[:, -1, :], ...)
    draft_tokens.append(dnext)
    cur = dnext
```

The last forward pass through the draft model is called with `cur = d_{g-1}`, which *produces* `d_g` as output but never feeds `d_g` itself into the model. So after the loop, the draft cache holds positions up to and including `d_{g-1}`, i.e. its length is `L0 + g - 1`, not `L0 + g`.

When the target accepts all `g` drafted tokens (`num_accepted == g`), the code calls:

```python
draft_model.rollback_kv_cache(L0 + num_accepted)   # = L0 + g
```

`rollback_kv_cache` slices with `k_cache[:, :, :n, :]`. Slicing to `n` on a tensor that is only `n-1` long **does not raise** — it silently returns the whole tensor. The draft cache therefore stays at `L0 + g - 1`, one token shorter than the running sequence implies.

Consequences in the next speculative round:
- the bonus token is fed to the draft model with a position index one too low (positions are derived from cache length in `forward`)
- `d_g` is missing entirely from the draft model's context

### Why the existing tests don't catch it

The target verifies every drafted token and corrects the first mismatch, so the **final output stays token-for-token identical** to plain greedy `generate()`. The identity test passes. Only the draft model's predictions degrade, which shows up exclusively as a depressed acceptance rate and worse speedup — i.e. it corrupts the measurements, not the correctness.

### Fix

Use the targeted approach, not the uniform one: after the drafting loop, **only when `num_accepted == g`**, run one extra draft forward pass over `d_g` to bring the draft cache up to `L0 + g` before the rollback. This pays the cost of one extra draft pass only in the full-acceptance case, rather than on every round.

(The alternative — feeding each drafted token through the draft model immediately so the cache is always `L0 + g` — is cleaner but pays that cost every round, which is the wrong trade at the observed acceptance rates.)

### Also add a guard

Make the silent-truncation failure mode impossible to repeat. In `rollback_kv_cache`:

```python
def rollback_kv_cache(self, n):
    for block in self.transformer.h:
        cur_len = block.attn.k_cache.size(2)
        assert cur_len >= n, (
            f"rollback to {n} requested but cache is only {cur_len} long; "
            f"rollback can only shorten a cache, never extend it"
        )
        block.attn.k_cache = block.attn.k_cache[:, :, :n, :]
        block.attn.v_cache = block.attn.v_cache[:, :, :n, :]
```

Keep this assert permanently — it is cheap and it is exactly what would have surfaced this bug immediately.

### New test required

Add a test that forces the specific scenario: **full acceptance → bonus token → at least one more speculative round**. A single-round test cannot catch this, since the corruption only manifests on the round *after* a fully-accepted block.

Because acceptance depends on the models actually agreeing, construct the scenario deterministically rather than hoping for it: use a draft and target that are guaranteed to agree (e.g. temporarily use identical weights via a test-only path, or a small handcrafted setup where greedy agreement is forced), with `max_new_tokens` large enough that at least two rounds run. Assert that after the fully-accepted round, `draft_model.transformer.h[0].attn.k_cache.size(2)` equals the expected `L0 + g`, and that the final output still matches plain greedy `generate()`.

Note: the existing draft-size guard (`assert n_draft < n_target`) will reject identical models, so the test needs a path around it — either a test-only keyword argument that skips the size assert, or two models of different size that are constructed to agree on the specific prompt. Pick whichever is less invasive; do not remove the guard from the production path.

---

## Problem 2: `alpha` as currently computed is not the conditional acceptance probability

### What's wrong

Current code:

```python
total_proposed += g
total_accepted += num_accepted
alpha = total_accepted / total_proposed
```

This computes the **accepted draft fraction**: accepted tokens divided by *all* proposed tokens. That is not the same quantity as the conditional acceptance probability α that appears in the analytical model.

If the true per-token conditional acceptance probability is a constant α, then the expected number of accepted tokens in a block of length γ is:

```
E[A] = α + α² + ... + α^γ
```

so the expectation of the currently-computed metric is:

```
E[A/γ] = (α + α² + ... + α^γ) / γ
```

which is strictly less than α for γ > 1, and **decreases monotonically with γ even when the true α is perfectly constant**. The observed pattern of "α declining with γ" in the preliminary measurements is therefore partly (possibly entirely) an artifact of the definition, not a property of the models.

This matters directly: the notebook feeds this number into

```
S(γ) = (1 - α^(γ+1)) / ((1 - α) · (1 + γc))
```

and plots measured speedup against that curve. Using the accepted-fraction where the formula expects the conditional probability makes the theoretical reference curve wrong.

### The root cause

Positions after the first rejection are **never tested** — the round stops at the first mismatch. Counting them in the denominator treats untested positions as failures. The denominator must count only positions that actually underwent the acceptance test.

### Fix

Track three distinct quantities in `generate_speculative`, and return all of them in the stats dict:

```python
# positions that actually reached the acceptance test this round:
# all accepted ones, plus the one that failed (if any).
# If the whole block was accepted, no position was rejected, so it's just num_accepted.
tested_this_round = num_accepted + (1 if num_accepted < g else 0)

total_tested += tested_this_round
total_accepted += num_accepted
total_proposed += g
accepted_per_round.append(num_accepted)
```

Return in stats:

| Key | Formula | Meaning |
|---|---|---|
| `alpha_conditional` | `total_accepted / total_tested` | estimate of the per-token conditional acceptance probability α — **this is what goes into the analytical formula** |
| `accepted_fraction` | `total_accepted / total_proposed` | the old metric; keep it, it's a legitimate descriptive measure of "how much draft work wasn't wasted" |
| `expected_accepted_per_round` | `mean(accepted_per_round)` | empirical E[A], directly comparable to the analytical `α(1-α^γ)/(1-α)` |
| `accepted_per_round` | the raw list | needed for variance/range reporting |
| `rounds` | `len(accepted_per_round)` | number of target forward passes, useful for sanity checks |

Keep `proposed`, `accepted`, and the existing `alpha` key for backward compatibility with any notebook cell already reading them, but have `alpha` be an explicit alias of `accepted_fraction` with a comment stating that it is **not** the quantity in the analytical model.

### Update `bench.ipynb` accordingly

- The theoretical curve in Figure 3 must use `alpha_conditional`, not `accepted_fraction`. This changes the plotted curve — that's the point of the fix.
- Add `alpha_conditional`, `accepted_fraction`, and `expected_accepted_per_round` to the metrics list in `benchmark_config_speculative`, all subject to the same median/range treatment.
- The existing determinism assert (alpha identical across repeats under greedy) should now apply to `alpha_conditional` as well — under greedy Phase A all three quantities are deterministic.
- Figure 4 should plot `alpha_conditional` vs. γ. Optionally add `accepted_fraction` as a second, visually distinct series in the same figure, since showing both makes the distinction concrete and is a genuinely interesting methodological point for the thesis.

### New analysis this enables

Add a cell that compares the **measured** `expected_accepted_per_round` against the **analytical** `α(1-α^γ)/(1-α)` computed from the measured `alpha_conditional`, per γ.

If they agree, the independence assumption behind the geometric-series derivation holds empirically. If measured E[A] systematically deviates, that is direct evidence that acceptances are correlated across positions within a block — which is a limitation the thesis already plans to state, but this turns it from an assumed caveat into a measured finding. Either outcome is a legitimate and reportable result; do not tune anything to force agreement.

---

## Order of work

1. Fix Problem 1 (draft cache) and add the `rollback_kv_cache` assert.
2. Add the multi-round full-acceptance regression test. Confirm it fails before the fix and passes after.
3. Fix Problem 2 (metric separation) in `model.py`.
4. Update `bench.ipynb` to use the corrected metrics and add the E[A] comparison cell.
5. Re-run the preliminary α sweep locally (toy models, pipeline check only) to confirm nothing crashes, then note clearly that the real numbers require a Colab GPU re-run — **all previously collected α numbers are invalid and must be re-measured after these fixes.**

Do not implement Phase B (stochastic acceptance) or int8 KV cache quantization; both remain out of scope.

---

## Resolution log (applied 2026-09-07)

Both problems verified against the actual code before fixing (not just the diagnosis above):

- **Problem 1** reproduced directly: a controlled repro script showed the draft cache landing at length `L0+gamma-1` instead of `L0+gamma` after one drafting round, and `rollback_kv_cache(L0+gamma)` silently no-op'ing on that short cache — exactly as predicted.
- **Problem 2** confirmed by reading the verify loop: it `break`s on first mismatch, so `total_proposed += g` counts positions the loop never reached. The math in this doc checks out: `(alpha+alpha^2+...+alpha^gamma)/gamma < alpha` for gamma>1, and decreases as gamma grows even under constant alpha.

Fixes applied in `mingpt/model.py`:
- `rollback_kv_cache` now asserts `cur_len >= n` before truncating.
- `generate_speculative` runs one extra draft forward pass over `d_g` (the last drafted token) immediately before rollback, but only when the round was a full acceptance (`correction is None`) — matching the targeted, not-every-round approach this doc specifies.
- `generate_speculative`'s `return_stats` output now includes `alpha_conditional`, `accepted_fraction`, `expected_accepted_per_round`, `tested`, and `rounds`, alongside the original `proposed`/`accepted`/`alpha` (kept as an alias of `accepted_fraction`) for backward compatibility.

New regression test: `TestSpeculativeDraftCacheInvariant` in `tests/test_speculative.py`, using a `copy.deepcopy` of the target as the draft (identical weights force deterministic full agreement every tested position, with independent cache tensors — unlike passing the same object as both target and draft, which would alias the two caches and defeat the point). Required a test-only `_skip_size_guard` kwarg on `generate_speculative` to bypass the draft-must-be-smaller assertion for this specific scenario. Confirmed failing on the pre-fix code (in fact failing *earlier* than expected, on the accepted==proposed premise itself — round 2's draft proposals disagreed with the target because round 1's corruption had already desynced the draft's context, which is itself a valid demonstration of the bug's downstream effect on measured acceptance) and passing after the fix. Full suite (8 tests) passes after the fix.

`bench.ipynb` updated: `run_once_speculative`/`benchmark_config_speculative`/`save_result_speculative` plumb through all the new stats fields (with a determinism assert on both `alpha` and `alpha_conditional`); the real sweep cell also picked up an unrelated but adjacent fix (`SPEC_PROMPT_LEN` computed as the true tokenizer-measured minimum across all 16 prompts, rather than a hardcoded 16 that several prompts couldn't actually reach); Figure 3's theoretical curve now uses `alpha_conditional`; Figure 4 plots `alpha_conditional` as the primary series with `accepted_fraction` overlaid as a visually distinct dotted series per category; Figure 5's table column renamed to `alpha_conditional`; a new E[A]-vs-analytical validation cell was added after Figure 4.

**All speculative-decoding α/speedup numbers collected before this fix are invalid and must be re-measured on Colab GPU.**
