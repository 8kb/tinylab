"""
BPE tokenizer, ported from nanochat's nanochat/tokenizer.py so tinylab produces identical token
ids without depending on nanochat itself -- see AGENTS.md for why it can't be a real dependency.
tinylab ships a committed default vocab (default_tokenizer/), so most job files never need to
train their own -- but tinylab.ops.tokenizer's `tokenizer` op can, via train_from_iterator/save
below (rustbpe + tiktoken, same as nanochat's own scripts/tok_train.py).
"""
import copy
import hashlib
import os
import pickle
import shutil
from importlib import resources

import tiktoken

# Named tokenizers live side by side under <base_dir>/tokenizers/<name>/. `default` is the one tinylab
# ships (default_tokenizer/), copied there on first use; every other name must already exist -- see
# get_tokenizer.
TOKENIZERS_DIR = "tokenizers"
DEFAULT_TOKENIZER_NAME = "default"

# Documents the special tokens baked into default_tokenizer/tokenizer.pkl -- not read at runtime
# (encode_special looks them up by string literal against the loaded tiktoken.Encoding directly),
# but useful as a single place a reader can see the full set without grepping.
SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]

# A rendered conversation longer than this is truncated (helps prevent OOMs on an unusually long
# conversation slipping into an SFT mixture). Also datacore/writer.py's row_capacity for the SFT
# dataset in practice matches sequence_len + 1, so this rarely binds first -- but it's the shared
# default both tinylab.tokenizer.RustBPETokenizer.render_conversation and tinylab.ops.prepare use,
# rather than two independent copies of the same number.
DEFAULT_MAX_TOKENS_PER_CONVERSATION = 2048

# tiktoken's own splitting regex, used only by train_from_iterator (a trained vocab's mergeable
# ranks bake in whatever pattern trained them; a *loaded* tokenizer's pat_str is already fixed).
# Deviates from GPT-4 in using \p{N}{1,2} instead of \p{N}{1,3} -- nanochat's own comment: "I
# didn't want to 'waste' too many tokens on numbers for smaller vocab sizes. I verified that 2 is
# the sweet spot for vocab size of 32K." Byte-identical to nanochat's own SPLIT_PATTERN, so a
# tokenizer retrained here on the same corpus reproduces nanochat's own token ids exactly.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference), trained with rustbpe. The bundled
    default vocab (default_tokenizer/) covers most use -- see from_directory / get_tokenizer below
    for loading it, train_from_iterator / save for training a new one."""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self._special_cache = {}
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """Trains a fresh vocab on text_iterator's documents -- ported from nanochat's
        nanochat/tokenizer.py's classmethod of the same name (rustbpe does the actual BPE merges;
        this wraps the result in a tiktoken.Encoding for fast inference). SPECIAL_TOKENS are never
        trained -- rustbpe only ever sees vocab_size - len(SPECIAL_TOKENS) ordinary tokens, and the
        specials are appended afterward in their fixed list order, so their ids are always the
        highest len(SPECIAL_TOKENS) ids in the vocab (e.g. "<|bos|>" = vocab_size - 9 by default).
        Imports rustbpe lazily: it's a training-only dependency, and tinylab.tokenizer is imported
        unconditionally by nearly everything, most of which never trains a tokenizer at all."""
        import rustbpe
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe", pat_str=tokenizer.get_pattern(),
            mergeable_ranks=mergeable_ranks, special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    def save(self, tokenizer_dir):
        """The from_directory-loadable half of train_from_iterator's output -- only self.enc (the
        tiktoken.Encoding) is pickled, same as nanochat's own save(). Doesn't write token_bytes.pt
        -- that's tinylab.ops.tokenizer's job, since it's derived from this tokenizer via
        token_byte_lengths(), not part of the tokenizer's own on-disk identity."""
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def encode_special(self, text):
        # Plain per-instance dict, not @lru_cache: an lru_cache on a bound method keys on `self`,
        # so it would pin every tokenizer instance alive for the process lifetime and share one
        # eviction pool across all of them. There are only 9 special tokens; a dict never evicts
        # and costs nothing extra.
        if text not in self._special_cache:
            self._special_cache[text] = self.enc.encode_single_token(text)
        return self._special_cache[text]

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)

    def decode_single_token_bytes(self, token_id):
        return self.enc.decode_single_token_bytes(token_id)

    def descriptor(self, name):
        """The opaque `tokenizer` block a checkpoint's model config carries (modelcore.ModelConfig.
        tokenizer), so a checkpoint says which tokenizer it needs: `name` is the spec a job used to
        select it (see resolve_tokenizer_dir), the rest is what identifies it. modelcore only
        carries this -- tinylab.checkpoints.build_model is what checks it."""
        return {"name": name, "fingerprint": self.fingerprint(), "vocab_size": self.get_vocab_size(),
                "special_tokens": sorted(self.get_special_tokens())}

    def fingerprint(self):
        """Content hash of the vocab (first 16 hex chars of sha256 over every token's bytes, in id
        order) -- identifies *what a token id means*, not the file it happens to be pickled as.
        Used to catch a model trained against a different tokenizer than the one loaded here,
        which would otherwise pass a vocab_size-only compatibility check and silently produce
        garbage -- see tinylab.checkpoints.build_model."""
        h = hashlib.sha256()
        for token_id in range(self.get_vocab_size()):
            h.update(self.decode_single_token_bytes(token_id))
        return h.hexdigest()[:16]

    def token_byte_lengths(self):
        """A list[int] of length vocab_size: the number of bytes for each token id, or 0 for a
        special token. This is the tokenizer's own optional half of datacore.tokenizer.Tokenizer's
        protocol -- datacore.DataManager.prepare calls this at prepare() time and persists the
        result alongside the dataset, so evaluate_bpb never needs a tokenizer directory at eval
        time. Special ids are zeroed regardless of what decode_single_token_bytes would report;
        every other token's length comes from its raw bytes, not a decode()-to-string first, which
        would corrupt a token that isn't valid standalone UTF-8."""
        special_ids = set(self.encode_special(s) for s in self.get_special_tokens())
        lengths = []
        for token_id in range(self.get_vocab_size()):
            if token_id in special_ids:
                lengths.append(0)
            else:
                lengths.append(len(self.decode_single_token_bytes(token_id)))
        return lengths

    def render_conversation(self, conversation, max_tokens=DEFAULT_MAX_TOKENS_PER_CONVERSATION):
        """Tokenize a single Chat conversation ("doc"). Returns (ids, mask): mask=1 for tokens the
        Assistant is expected to train on."""
        messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        if messages[0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]

        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
            content = message["content"]
            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    add_tokens(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        """Used by benchcore's generative chat tasks (GSM8K, HumanEval): primes the Assistant for
        a completion by popping the last (Assistant) message and appending assistant_start."""
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()
        ids, _mask = self.render_conversation(conversation)
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids


# -----------------------------------------------------------------------------
# tinylab-specific convenience functions

def _bundled_default_tokenizer_dir():
    """The path to tinylab's own committed default_tokenizer/ (tokenizer.pkl + token_bytes.pt),
    ported byte-for-byte from nanochat/nanochat/default_tokenizer/ -- so a fresh tinylab checkout
    can train/chat immediately with no tok_train step, and produces token ids identical to
    nanochat's own default tokenizer."""
    return str(resources.files("tinylab") / "default_tokenizer")


def resolve_tokenizer_dir(spec=None, *, base_dir=None):
    """Which directory a job's "tokenizer" value names: None -> the default tokenizer; a bare name ->
    <base_dir>/tokenizers/<name>/; anything containing a path separator -> that path (the same
    "a separator means a path" rule as model_config; tinylab.job has already made a relative one
    absolute against the job file's own directory by the time it gets here)."""
    from tinylab.runtime import get_base_dir
    if spec is None:
        spec = DEFAULT_TOKENIZER_NAME
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"tokenizer must be a non-empty name or path, got {spec!r}")
    if "/" in spec or os.sep in spec:
        return os.path.abspath(os.path.expanduser(spec))
    if spec in (".", "..") or "\0" in spec:
        raise ValueError(f"tokenizer name {spec!r} is not a valid directory name")
    return os.path.join(base_dir or get_base_dir(), TOKENIZERS_DIR, spec)


def get_tokenizer(base_dir=None, tokenizer=None):
    """Loads the tokenizer `tokenizer` names (see resolve_tokenizer_dir; None -> the default).
    Only the default one is ever materialized for you -- tinylab's bundled vocab is copied into
    <base_dir>/tokenizers/default/ on first use (mirrors nanochat's convention of a repo-committed
    tokenizer, but does the copy in Python instead of requiring a shell step -- tinylab has no
    shell runners). Any other name that doesn't exist raises: silently handing back the default
    vocab under a different name would pass every fingerprint check, since they would be
    fingerprint-identical."""
    tokenizer_dir = resolve_tokenizer_dir(tokenizer, base_dir=base_dir)
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(pickle_path):
        if tokenizer is not None and tokenizer != DEFAULT_TOKENIZER_NAME:
            raise FileNotFoundError(
                f"tokenizer {tokenizer!r} not found at {tokenizer_dir} -- train one with a "
                f"\"tokenizer\" step (its \"output\" key names where to write), or point "
                f"\"tokenizer\" at an existing directory."
            )
        os.makedirs(tokenizer_dir, exist_ok=True)
        bundled_dir = _bundled_default_tokenizer_dir()
        shutil.copy(os.path.join(bundled_dir, "tokenizer.pkl"), pickle_path)
        token_bytes_src = os.path.join(bundled_dir, "token_bytes.pt")
        if os.path.exists(token_bytes_src):
            shutil.copy(token_bytes_src, os.path.join(tokenizer_dir, "token_bytes.pt"))
    return RustBPETokenizer.from_directory(tokenizer_dir)
