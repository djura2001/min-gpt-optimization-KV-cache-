"""
Correctness (identity) tests for generate_speculative, Phase A (greedy).

Per CLAUDE.md, this is the gate that must pass before any speculative-decoding
result is trusted for benchmarking: with do_sample=False, generate_speculative
must produce token-for-token identical output to plain generate() on the target
model, for any gamma and for any combination of vanilla/cached target and draft.
This only checks implementation correctness, not accept rate (alpha) -- these
toy models are untrained/random so alpha is meaningless here by construction.
"""

import unittest

import torch

from mingpt.model import GPT
from mingpt.utils import set_seed

VOCAB_SIZE = 100
BLOCK_SIZE = 64
PROMPT_LEN = 10
MAX_NEW_TOKENS = 15
GAMMAS = [1, 2, 3, 4, 7]  # 7 > remaining budget on the last round, exercises the overshoot guard


def make_toy(model_type, vanilla, seed=1234):
    set_seed(seed)
    cfg = GPT.get_default_config()
    cfg.model_type = model_type
    cfg.vocab_size = VOCAB_SIZE
    cfg.block_size = BLOCK_SIZE
    m = GPT(cfg, vanilla=vanilla)
    m.eval()
    return m


class TestSpeculativeIdentityToy(unittest.TestCase):
    """All 4 vanilla/cache combinations, random untrained toy models."""

    def _check(self, target_vanilla, draft_vanilla):
        for gamma in GAMMAS:
            with self.subTest(target_vanilla=target_vanilla, draft_vanilla=draft_vanilla, gamma=gamma):
                target = make_toy('gpt-micro', target_vanilla)
                draft = make_toy('gpt-nano', draft_vanilla)

                torch.manual_seed(0)
                prompt = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))

                ref = target.generate(prompt.clone(), max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                spec = target.generate_speculative(
                    draft, prompt.clone(), max_new_tokens=MAX_NEW_TOKENS, gamma=gamma, do_sample=False
                )
                self.assertEqual(ref.shape, spec.shape)
                self.assertTrue(torch.equal(ref, spec))

    def test_cached_target_cached_draft(self):
        self._check(target_vanilla=False, draft_vanilla=False)

    def test_vanilla_target_cached_draft(self):
        self._check(target_vanilla=True, draft_vanilla=False)

    def test_cached_target_vanilla_draft(self):
        self._check(target_vanilla=False, draft_vanilla=True)

    def test_vanilla_target_vanilla_draft(self):
        self._check(target_vanilla=True, draft_vanilla=True)


class TestSpeculativeOvershootGuard(unittest.TestCase):
    """gamma >> remaining budget must never emit more than max_new_tokens tokens,
    even when every drafted token is accepted (the bonus-token-drop branch)."""

    def test_no_overshoot(self):
        target = make_toy('gpt-micro', vanilla=False)
        draft = make_toy('gpt-nano', vanilla=False)

        torch.manual_seed(0)
        prompt = torch.randint(0, VOCAB_SIZE, (1, 6))

        for max_new_tokens in [0, 1, 2, 3, 4, 5]:
            for gamma in [8, 16]:
                with self.subTest(max_new_tokens=max_new_tokens, gamma=gamma):
                    ref = target.generate(prompt.clone(), max_new_tokens=max_new_tokens, do_sample=False)
                    spec = target.generate_speculative(
                        draft, prompt.clone(), max_new_tokens=max_new_tokens, gamma=gamma, do_sample=False
                    )
                    self.assertEqual(ref.shape, spec.shape)
                    self.assertTrue(torch.equal(ref, spec))


class TestSpeculativeDraftSizeGuard(unittest.TestCase):
    """generate_speculative must refuse a draft that isn't smaller than the target."""

    def test_rejects_equal_or_larger_draft(self):
        target = make_toy('gpt-micro', vanilla=False)
        same_size_draft = make_toy('gpt-micro', vanilla=False)

        torch.manual_seed(0)
        prompt = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))

        with self.assertRaises(AssertionError):
            target.generate_speculative(same_size_draft, prompt, max_new_tokens=5, gamma=2, do_sample=False)


class TestSpeculativeDraftCacheInvariant(unittest.TestCase):
    """Regression test for the draft-KV-cache-one-token-short-after-full-acceptance
    bug (prompt_alpha_and_cache_fixes.md, Problem 1). A single round can't expose
    this: the drift only shows up on the round AFTER a fully-accepted block, so this
    forces multiple rounds with guaranteed full agreement on every round.

    Uses a deepcopy of the target as the draft (identical weights -> deterministic
    full acceptance at every tested position, since draft and target compute the
    exact same greedy argmax given the same prefix) with independent cache tensors
    (unlike passing the same object as both target and draft, which would alias the
    two caches together and defeat the point of isolating the draft's own bookkeeping).
    """

    def test_draft_cache_length_after_full_acceptance_round(self):
        import copy

        target = make_toy('gpt-micro', vanilla=False)
        draft = copy.deepcopy(target)

        torch.manual_seed(0)
        prompt = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))
        gamma = 3
        # 9 = 1 (prefill) + 4 (round 1: 3 accepted + bonus) + 4 (round 2: 3 accepted + bonus),
        # chosen so both rounds end normally (bonus token committed) rather than hitting the
        # overshoot-drop branch -- that branch commits only the accepted prefix with no bonus,
        # so the draft cache legitimately ends up fully caught up (lag 0) instead of lag-1 there,
        # which would make the generic invariant checked below the wrong expectation.
        max_new_tokens = 9

        ref = target.generate(prompt.clone(), max_new_tokens=max_new_tokens, do_sample=False)
        spec, stats = target.generate_speculative(
            draft, prompt.clone(), max_new_tokens=max_new_tokens, gamma=gamma,
            do_sample=False, return_stats=True, _skip_size_guard=True,
        )

        # output correctness must hold regardless of this bug -- the target verifies
        # independently of the draft's (possibly corrupted) internal state
        self.assertTrue(torch.equal(ref, spec))

        # this test's own premise: identical draft/target must fully accept every
        # proposed token (no disagreement is possible with identical greedy models)
        self.assertEqual(stats['accepted'], stats['proposed'],
                          "test setup assumption violated: identical draft/target should "
                          "fully accept every proposed token")

        # the actual regression check: the draft cache must maintain the same
        # "lags idx by exactly 1" invariant the target cache and generate() maintain,
        # not drift short after full-acceptance rounds
        final_len = spec.size(1)
        draft_cache_len = draft.transformer.h[0].attn.k_cache.size(2)
        self.assertEqual(draft_cache_len, final_len - 1,
                          f"draft cache length {draft_cache_len} != expected {final_len - 1} "
                          f"-- draft cache is drifting short after full-acceptance rounds")


class TestSpeculativeIdentityGPT2(unittest.TestCase):
    """Same identity check on real trained weights (gpt2 draft / gpt2-medium target).
    Requires the gpt2 and gpt2-medium checkpoints to be resolvable via huggingface_hub
    (already cached locally as of writing this test)."""

    @classmethod
    def setUpClass(cls):
        cls.target = GPT.from_pretrained('gpt2-medium', vanilla=False).eval()
        cls.draft = GPT.from_pretrained('gpt2', vanilla=False).eval()
        from mingpt.bpe import BPETokenizer
        cls.tokenizer = BPETokenizer()

    def test_identity_real_weights(self):
        prompt = self.tokenizer('Michael Jordan is a')
        for gamma in [1, 3, 5]:
            with self.subTest(gamma=gamma):
                ref = self.target.generate(prompt.clone(), max_new_tokens=15, do_sample=False)
                spec = self.target.generate_speculative(
                    self.draft, prompt.clone(), max_new_tokens=15, gamma=gamma, do_sample=False
                )
                self.assertTrue(torch.equal(ref, spec))


if __name__ == '__main__':
    unittest.main()
