"""
Inference engine: KV-cached generation plus the calculator tool-use state machine. Ported from
nanochat's nanochat/engine.py, dropping only its __main__ equivalence-testing harness. Everything
works over token id sequences -- the Engine knows nothing about tokenization beyond the handful of
special tokens it needs for tool use.

The tool-use decode loop itself (RowState, the forced-token deque, terminal-token detection, the
tool start/end state machine) now lives in modelcore.generate.generate_with_tools/collect_batch --
nanochat's engine.py carried an identical copy of this whole file. What stays here is everything
actually specific to this tool and this chat format: use_calculator (the eval() sandbox), and
resolving this repo's own special-token names to ids for the ToolSpec/terminal_ids
generate_with_tools takes.

Engine.generate_batch satisfies benchcore.protocols.Generator unmodified, so the same instance
tinylab's `chat` command drives interactively is what tinylab.ops.bench hands to
BenchManager.chat/chat_suite for GSM8K/HumanEval scoring.
"""
import signal
import warnings
from contextlib import contextmanager

from modelcore import ModelManager
from modelcore.generate import ToolSpec, collect_batch, generate_with_tools

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


class Engine:

    def __init__(self, model, tokenizer, manager=None):
        self.model = model
        self.tokenizer = tokenizer  # needed for tool use
        self.manager = manager or ModelManager()

    def _run_calculator(self, captured_tokens):
        """ToolSpec.run for the python_start/python_end tool: decode the captured tokens, hand
        the resulting expression to use_calculator, re-encode the result (or return None -- wrong
        calculator usage is not fatal, generate_with_tools injects nothing in that case)."""
        expr = self.tokenizer.decode(captured_tokens)
        result = use_calculator(expr)
        if result is None:
            return None
        return self.tokenizer.encode(str(result))

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Single prefill, then decode num_samples rows from a shared KV cache. Yields
        (token_column, token_masks) per step: mask=0 where a token was tool-forced, 1 if sampled."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"

        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        tool = ToolSpec(python_start, python_end, output_start, output_end, run=self._run_calculator)
        yield from generate_with_tools(
            self.model, self.manager, tokens, num_samples=num_samples, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, seed=seed,
            terminal_ids={assistant_end, bos}, tools=[tool],
        )

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """Non-streaming batch generation. Returns (results, masks): each a list of num_samples
        token-id lists. Terminal tokens (assistant_end, bos) are excluded."""
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        stream = self.generate(tokens, num_samples, **kwargs)
        return collect_batch(stream, {assistant_end, bos}, tokens, num_samples)
