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

        for max_new_tokens in [1, 2, 3, 4, 5]:
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
