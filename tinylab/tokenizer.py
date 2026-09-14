"""
BPE tokenizer, ported from nanochat's nanochat/tokenizer.py so tinylab produces identical token
ids without depending on nanochat itself -- see AGENTS.md for why it can't be a real dependency.
Inference-only: tinylab ships a committed vocab (default_tokenizer/) and has no `tok_train` op, so
the rustbpe-based training path is dropped -- kept: encode/decode, the special-token and
byte-length helpers evaluate_bpb needs, and the two conversation-rendering entry points (chat SFT
data prep, benchcore's generative eval).
"""
import copy
import hashlib
import os
import pickle
import shutil
from importlib import resources

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


class RustBPETokenizer:
    """Light wrapper around tiktoken. The vocab itself is trained once, offline (see nanochat's
    own tok_train.py, since tinylab has no equivalent) and loaded from a pickled tiktoken.Encoding
    -- see from_directory / get_tokenizer below."""

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


def get_tokenizer(base_dir=None):
    """Loads the tokenizer from <base_dir>/tokenizer/, copying tinylab's bundled default vocab
    there on first use if nothing has been trained yet (mirrors nanochat's convention of a
    repo-committed tokenizer, but does the copy in Python instead of requiring a shell step --
    tinylab has no shell runners)."""
    from tinylab.runtime import get_base_dir
    base_dir = base_dir or get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(pickle_path):
        os.makedirs(tokenizer_dir, exist_ok=True)
        bundled_dir = _bundled_default_tokenizer_dir()
        shutil.copy(os.path.join(bundled_dir, "tokenizer.pkl"), pickle_path)
        token_bytes_src = os.path.join(bundled_dir, "token_bytes.pt")
        if os.path.exists(token_bytes_src):
            shutil.copy(token_bytes_src, os.path.join(tokenizer_dir, "token_bytes.pt"))
    return RustBPETokenizer.from_directory(tokenizer_dir)
