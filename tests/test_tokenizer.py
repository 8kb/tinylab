"""Tests for tinylab.tokenizer: the bundled default_tokenizer/ loads, fingerprint() is stable,
token_byte_lengths() zeroes special ids, and render_conversation's mask placement."""
import os

import pytest

from tinylab.tokenizer import DEFAULT_TOKENIZER_NAME, RustBPETokenizer, get_tokenizer, resolve_tokenizer_dir


def test_get_tokenizer_copies_bundled_default_on_first_use(base_dir):
    tok = get_tokenizer()
    assert os.path.exists(os.path.join(base_dir, "tokenizers", "default", "tokenizer.pkl"))
    assert tok.get_vocab_size() > 256


# -- named tokenizers -------------------------------------------------------------------------

def test_resolve_tokenizer_dir_name_path_and_default(base_dir):
    assert resolve_tokenizer_dir(None) == os.path.join(base_dir, "tokenizers", DEFAULT_TOKENIZER_NAME)
    assert resolve_tokenizer_dir("bpe32k") == os.path.join(base_dir, "tokenizers", "bpe32k")
    # a value with a separator is a path, not a name -- absolute is kept, ~ is expanded
    assert resolve_tokenizer_dir("/opt/toks/mine") == "/opt/toks/mine"
    assert resolve_tokenizer_dir("~/toks/mine") == os.path.expanduser("~/toks/mine")
    assert os.path.isabs(resolve_tokenizer_dir("./toks/mine"))


@pytest.mark.parametrize("bad", ["", "   ", ".", "..", 3, ["a"]])
def test_resolve_tokenizer_dir_rejects_nonsense(base_dir, bad):
    with pytest.raises(ValueError):
        resolve_tokenizer_dir(bad)


def test_only_the_default_tokenizer_is_materialized_for_you(base_dir):
    """A missing *named* tokenizer must raise, not quietly copy the bundled vocab under that name --
    the two would be fingerprint-identical, so no downstream identity check could catch it."""
    with pytest.raises(FileNotFoundError, match="mine.*tokenizer.*step"):
        get_tokenizer(tokenizer="mine")
    assert not os.path.exists(os.path.join(base_dir, "tokenizers", "mine"))
    get_tokenizer(tokenizer="default")  # the default name is still self-materializing
    assert os.path.exists(os.path.join(base_dir, "tokenizers", "default", "tokenizer.pkl"))


def test_named_tokenizers_coexist_and_a_path_can_point_anywhere(base_dir, tmp_path):
    default = get_tokenizer()
    other = str(tmp_path / "elsewhere" / "toks")
    default.save(other)
    by_path = get_tokenizer(tokenizer=other)
    assert by_path.fingerprint() == default.fingerprint()
    default.save(os.path.join(base_dir, "tokenizers", "copy"))
    assert get_tokenizer(tokenizer="copy").fingerprint() == default.fingerprint()


def test_descriptor_identifies_the_tokenizer(base_dir):
    tok = get_tokenizer()
    d = tok.descriptor("bpe32k")
    assert d["name"] == "bpe32k" and d["fingerprint"] == tok.fingerprint() and d["vocab_size"] == tok.get_vocab_size()
    assert "<|bos|>" in d["special_tokens"] and d["special_tokens"] == sorted(d["special_tokens"])


def test_fingerprint_is_stable_across_loads(base_dir):
    tok1 = get_tokenizer()
    tok2 = RustBPETokenizer.from_directory(os.path.join(base_dir, "tokenizers", "default"))
    assert tok1.fingerprint() == tok2.fingerprint()
    assert len(tok1.fingerprint()) == 16


def test_encode_special_cache_is_per_instance(base_dir):
    """Regression test for the @lru_cache-on-a-bound-method bug: two tokenizer instances must not
    share (or evict from) one cache, and both must still resolve every special token correctly."""
    tok1 = get_tokenizer()
    tok2 = get_tokenizer()
    assert tok1 is not tok2
    for name in tok1.get_special_tokens():
        assert tok1.encode_special(name) == tok2.encode_special(name)
    assert tok1._special_cache is not tok2._special_cache


def test_token_byte_lengths_zeroes_special_ids_and_normal_tokens_are_nonzero(base_dir):
    tok = get_tokenizer()
    lengths = tok.token_byte_lengths()
    assert len(lengths) == tok.get_vocab_size()
    special_ids = {tok.encode_special(s) for s in tok.get_special_tokens()}
    for token_id in special_ids:
        assert lengths[token_id] == 0
    # every non-special token must have encoded to at least one real byte
    normal_ids = [i for i in range(tok.get_vocab_size()) if i not in special_ids]
    assert normal_ids and all(lengths[i] > 0 for i in normal_ids)


def test_render_conversation_masks_only_assistant_tokens(base_dir):
    tok = get_tokenizer()
    conversation = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}
    ids, mask = tok.render_conversation(conversation)
    assert len(ids) == len(mask)
    assistant_start = tok.encode_special("<|assistant_start|>")
    start_idx = ids.index(assistant_start)
    assert all(m == 0 for m in mask[:start_idx])  # bos + user turn: unsupervised
    assert any(m == 1 for m in mask[start_idx:])  # assistant turn: at least some supervised tokens


def test_render_conversation_rejects_empty_messages(base_dir):
    tok = get_tokenizer()
    try:
        tok.render_conversation({"messages": []})
        assert False, "expected an AssertionError before any message is indexed"
    except AssertionError as e:
        assert "less than 1 message" in str(e)


def test_render_for_completion_pops_last_assistant_message(base_dir):
    tok = get_tokenizer()
    conversation = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]}
    ids = tok.render_for_completion(conversation)
    assert ids[-1] == tok.encode_special("<|assistant_start|>")
