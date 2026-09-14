"""
Inference engine: KV-cached generation plus the calculator tool-use state machine. Ported from
nanochat's nanochat/engine.py, dropping only its __main__ equivalence-testing harness. Everything
works over token id sequences -- the Engine knows nothing about tokenization beyond the handful of
special tokens it needs for tool use.

Engine.generate_batch satisfies benchcore.protocols.Generator unmodified, so the same instance
tinylab's `chat` command drives interactively is what tinylab.ops.bench hands to
BenchManager.chat/chat_suite for GSM8K/HumanEval scoring.
"""
import signal
import warnings
from collections import deque
from contextlib import contextmanager

import torch
from modelcore import ModelManager
from modelcore.generate import sample_next_token

# -----------------------------------------------------------------------------
# Calculator tool helpers


@contextmanager
def _timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def _eval_with_timeout(formula, max_time=3):
    # _timeout's own try/finally already guarantees signal.alarm(0) runs on any exit path
    # (including the SIGALRM-raised one below), so this except only needs to swallow it.
    try:
        with _timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        return None


def use_calculator(expr):
    """Evaluate a Python expression safely: pure math expressions, or a whitelisted .count()
    string operation. Anything else returns None (ignored -- wrong calculator usage is not fatal)."""
    expr = expr.replace(",", "")
    if all(x in "0123456789*+-/.() " for x in expr):
        if "**" in expr:  # disallow power operator
            return None
        return _eval_with_timeout(expr)
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all(x in allowed_chars for x in expr):
        return None
    dangerous_patterns = ["__", "import", "exec", "eval", "compile", "open", "file",
                          "input", "raw_input", "globals", "locals", "vars", "dir",
                          "getattr", "setattr", "delattr", "hasattr"]
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None
    if ".count(" not in expr:
        return None
    return _eval_with_timeout(expr)


class RowState:
    """Per-row state tracking during generation."""
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False


class Engine:

    def __init__(self, model, tokenizer, manager=None):
        self.model = model
        self.tokenizer = tokenizer  # needed for tool use
        self.manager = manager or ModelManager()

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Single prefill, then decode num_samples rows from a shared KV cache. Yields
        (token_column, token_masks) per step: mask=0 where a token was tool-forced, 1 if sampled."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        decoder = self.manager.new_decoder(self.model, tokens, num_samples=num_samples, max_tokens=max_tokens, device=device)
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(decoder.logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1
            decoder.step(token_column)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """Non-streaming batch generation. Returns (results, masks): each a list of num_samples
        token-id lists. Terminal tokens (assistant_end, bos) are excluded."""
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks
