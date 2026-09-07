"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttentionVanilla(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        T_q = q.size(2)
        T_k = k.size(2)
        att = att.masked_fill(self.bias[:, :, T_k - T_q:T_k, :T_k] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y,k,v,q
    
class CausalSelfAttention(CausalSelfAttentionVanilla):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.k_cache = None
        self.v_cache = None  

    def forward(self, x, K_cache = None, V_cache = None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.k_cache is None or self.v_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
    
            self.k_cache = torch.cat([self.k_cache, k], dim = 2)
            self.v_cache = torch.cat([self.v_cache, v], dim = 2)
        k = self.k_cache
        v = self.v_cache


        # causal self-attention; Self-attend: (B, nh, Tq, hs) x (B, nh, hs, Tk) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # When keys include cached previous tokens, k.size(2) may be > T.
        # We need the bias rows corresponding to the absolute positions of the queries,
        # which are the last `T` rows relative to the current cached key length.
        T_q = q.size(2)
        T_k = k.size(2)
        att = att.masked_fill(self.bias[:, :, T_k - T_q:T_k, :T_k] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs), where Tq == T
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y,k,v,q


        

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, vanilla):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if vanilla == False:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = CausalSelfAttentionVanilla(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        att_x, k, v, q = self.attn(self.ln_1(x))
        x = x + att_x
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config, vanilla=False):
        super().__init__()
        self.vanilla = vanilla
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config, self.vanilla) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.num_layers = config.n_layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type ,vanilla = False):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config, vanilla)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Koristi samo ključeve koji postoje u HuggingFace checkpointu
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        print(f"MinGPT model has: {len(sd)} parameters")
        print(f"HuggingFace checkpoint has: {len(sd_hf)} parameters")
        print(f"Keys to copy: {len(keys)}")
        
        # Kopiraj samo parametre koji postoje u oba modela
        for k in keys:
            if k not in sd:
                print(f"Warning: {k} not in minGPT model, skipping")
                continue
                
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    def reset_kv_cache(self):
        for block in self.transformer.h:
            att = block.attn
            att.k_cache = None
            att.v_cache = None

    def rollback_kv_cache(self, n):
        for block in self.transformer.h:
            cur_len = block.attn.k_cache.size(2)
            assert cur_len >= n, (
                f"rollback to {n} requested but cache is only {cur_len} long; "
                f"rollback can only shorten a cache, never extend it"
            )
            block.attn.k_cache = block.attn.k_cache[:, :, :n, :]
            block.attn.v_cache = block.attn.v_cache[:, :, :n, :]

    @staticmethod
    def _count_params(model):
        return sum(p.numel() for p in model.parameters())

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, kv_cached, targets=None):
        device = idx.device
        b, t = idx.size()
        past_len = 0
        if kv_cached and self.transformer.h[0].attn.k_cache is not None:
            past_len = self.transformer.h[0].attn.k_cache.size(2)
        assert past_len + t <= self.block_size, (
            f"KV cache exceeded block_size ({past_len + t} > {self.block_size}); "
            f"sliding window is not implemented, this is a documented limitation"
        )
        pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if self.vanilla == False:
            self.reset_kv_cache()
            #PREFILL
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _  = self(idx_cond, False)

            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            max_new_tokens = max_new_tokens - 1

        #DECODING

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            # When using KV-caching we pass only the newly generated token (`idx_next`);
            # its position is derived inside forward() from the cache length.
            if self.vanilla == False:
                logits, _ = self(idx_next, True)
            else:
                logits, _ = self(idx, False)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @staticmethod
    def _apply_temp_topk(logits, temperature, do_sample, top_k):
        # logits: (1, vocab). Applies temperature/top_k the same way generate() does,
        # returns the greedy (do_sample=False) or sampled next-token id.
        logits = logits.clone() / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            return torch.multinomial(probs, num_samples=1)
        _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next

    @torch.no_grad()
    def generate_speculative(self, draft_model, idx, max_new_tokens, gamma,
                              temperature=1.0, do_sample=False, top_k=None, return_stats=False,
                              _skip_size_guard=False):
        """
        Speculative decoding, Phase A (greedy). `self` is the target model.
        Draft proposes `gamma` tokens (with its own KV cache if draft_model.vanilla is
        False, else a full recompute each step); target verifies all gamma+1 positions
        in a single forward pass and accepts the greedy prefix that agrees with the
        draft, replacing the first disagreement (or emitting a bonus token if the whole
        draft was accepted). Batch size 1 only. Stochastic acceptance (Phase B) is not
        implemented; do_sample must be False here.

        If return_stats is True, returns (idx, stats) where stats is a dict; see the
        `return_stats` block at the end of this method for the exact keys and what each
        one means (in particular: 'alpha' is kept only for backward compatibility and is
        NOT the conditional acceptance probability the analytical speedup formula wants
        -- use 'alpha_conditional' for that).

        `_skip_size_guard` is test-only: it bypasses the draft-smaller-than-target
        assertion so a test can force deterministic full agreement (e.g. an identical
        deepcopy as the draft) to exercise multi-round behavior without depending on
        real model disagreement. Never set this outside tests.
        """
        assert idx.size(0) == 1, "speculative decoding only supports batch size 1"
        assert not do_sample, "stochastic acceptance (Phase B) is not implemented yet"
        assert gamma >= 1
        if not _skip_size_guard:
            n_target, n_draft = self._count_params(self), self._count_params(draft_model)
            assert n_draft < n_target, (
                f"draft model ({n_draft/1e6:.1f}M params) is not smaller than "
                f"target ({n_target/1e6:.1f}M params)"
            )

        if not self.vanilla:
            self.reset_kv_cache()
        if not draft_model.vanilla:
            draft_model.reset_kv_cache()

        # --- step 0: target prefill over the prompt produces the first token, exactly
        # as generate() would; draft prefill only needs to populate its own cache up to
        # (but not including) that first generated token, maintaining the invariant
        # that a cached model's KV cache always lags the running sequence by one token ---
        prompt_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
        logits, _ = self(prompt_cond, not self.vanilla)
        idx_next = self._apply_temp_topk(logits[:, -1, :], temperature, do_sample, top_k)
        idx = torch.cat((idx, idx_next), dim=1)
        max_new_tokens -= 1

        if not draft_model.vanilla and max_new_tokens > 0:
            prompt_cond_d = idx[:, :-1]
            if prompt_cond_d.size(1) > draft_model.block_size:
                prompt_cond_d = prompt_cond_d[:, -draft_model.block_size:]
            if prompt_cond_d.size(1) > 0:
                draft_model(prompt_cond_d, True)

        # --- speculative rounds ---
        total_proposed = 0
        total_accepted = 0
        total_tested = 0
        accepted_per_round = []
        while max_new_tokens > 0:
            g = min(gamma, max_new_tokens)
            L0 = idx.size(1)

            # draft proposes g tokens
            draft_tokens = []
            cur = idx[:, -1:]
            for _ in range(g):
                if draft_model.vanilla:
                    full = torch.cat([idx] + draft_tokens, dim=1) if draft_tokens else idx
                    if full.size(1) > draft_model.block_size:
                        full = full[:, -draft_model.block_size:]
                    dlogits, _ = draft_model(full, False)
                else:
                    dlogits, _ = draft_model(cur, True)
                dnext = self._apply_temp_topk(dlogits[:, -1, :], temperature, do_sample, top_k)
                draft_tokens.append(dnext)
                cur = dnext
            draft_tokens = torch.cat(draft_tokens, dim=1)  # (1, g)

            # target verifies all g+1 positions (last committed token + g draft tokens)
            # in a single forward pass
            if self.vanilla:
                full_seq = torch.cat([idx, draft_tokens], dim=1)
                if full_seq.size(1) > self.block_size:
                    full_seq = full_seq[:, -self.block_size:]
                vlogits, _ = self(full_seq, False)
                verify_logits = vlogits[:, -(g + 1):, :]
            else:
                verify_in = torch.cat([idx[:, -1:], draft_tokens], dim=1)
                verify_logits, _ = self(verify_in, True)

            num_accepted = 0
            correction = None
            for i in range(g):
                pick = self._apply_temp_topk(verify_logits[:, i, :], temperature, do_sample, top_k)
                if pick.item() == draft_tokens[0, i].item():
                    num_accepted += 1
                else:
                    correction = pick
                    break

            if correction is None:
                extra = self._apply_temp_topk(verify_logits[:, g, :], temperature, do_sample, top_k)  # bonus token
            else:
                extra = correction

            # Positions that actually underwent the accept/reject test above: all
            # accepted ones, plus the one that failed (if the round ended in a
            # rejection rather than full acceptance). Positions after a rejection are
            # never reached by the loop, so they must NOT be counted as tested -- doing
            # so is exactly what made the old `alpha` an underestimate of the true
            # per-token conditional acceptance probability, worsening as gamma grows
            # even when that probability is constant (see prompt_alpha_and_cache_fixes.md).
            tested_this_round = num_accepted + (1 if num_accepted < g else 0)
            total_tested += tested_this_round
            total_proposed += g
            total_accepted += num_accepted
            accepted_per_round.append(num_accepted)

            took = num_accepted + 1
            if took > max_new_tokens:
                # only reachable when g == max_new_tokens and every draft token was
                # accepted; drop the bonus token so we don't overshoot the requested length
                took = max_new_tokens
                idx = torch.cat([idx, draft_tokens[:, :num_accepted]], dim=1)
            else:
                idx = torch.cat([idx, draft_tokens[:, :num_accepted], extra], dim=1)
            max_new_tokens -= took

            if not draft_model.vanilla and correction is None:
                # Full acceptance: the drafting loop above never fed d_g (the last
                # drafted token) back into the draft model -- its final iteration used
                # d_{g-1} as input and only ever READ d_g as output. So at this point
                # the draft cache is exactly one token short of L0 + g. Left alone,
                # rollback_kv_cache(L0 + num_accepted) below would silently no-op short
                # (n > actual length), permanently desyncing the draft's cache from the
                # running sequence -- corrupting its future proposals (and therefore the
                # measured accept rate) while never affecting final output correctness,
                # since the target verifies independently. Sync it before rolling back.
                draft_model(draft_tokens[:, -1:], True)
            if not self.vanilla:
                self.rollback_kv_cache(L0 + num_accepted)
            if not draft_model.vanilla:
                draft_model.rollback_kv_cache(L0 + num_accepted)

        if return_stats:
            # accepted_fraction: the old 'alpha' -- accepted / ALL proposed tokens, including
            # ones a rejection meant were never actually tested. A legitimate descriptive
            # measure of "how much draft work wasn't wasted", but NOT the quantity the
            # analytical speedup formula (1-a^(g+1))/((1-a)(1+g*c)) expects.
            accepted_fraction = total_accepted / total_proposed if total_proposed > 0 else float('nan')
            # alpha_conditional: accepted / TESTED tokens only -- an estimate of the true
            # per-token conditional acceptance probability. Use this one for the analytical
            # formula and for Figure 3/4. See prompt_alpha_and_cache_fixes.md, Problem 2.
            alpha_conditional = total_accepted / total_tested if total_tested > 0 else float('nan')
            expected_accepted_per_round = (
                sum(accepted_per_round) / len(accepted_per_round) if accepted_per_round else float('nan')
            )
            stats = {
                'proposed': total_proposed,
                'accepted': total_accepted,
                'tested': total_tested,
                'alpha_conditional': alpha_conditional,
                'accepted_fraction': accepted_fraction,
                'expected_accepted_per_round': expected_accepted_per_round,
                'accepted_per_round': accepted_per_round,
                'rounds': len(accepted_per_round),
                # kept only for backward compatibility with existing call sites; this is
                # accepted_fraction, NOT the analytical model's alpha -- see above.
                'alpha': accepted_fraction,
            }
            return idx, stats
        return idx

