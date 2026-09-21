"""Tests for tinylab.checkpoints.find_last_step: picks the highest step from model_NNNNNN.pt
filenames in a checkpoint directory."""
import pytest

from tinylab.checkpoints import find_last_step


def test_finds_the_highest_step(tmp_path):
    for step in (0, 10, 5):
        (tmp_path / f"model_{step:06d}.pt").touch()
    assert find_last_step(str(tmp_path)) == 10


def test_ignores_non_matching_files(tmp_path):
    (tmp_path / "model_000003.pt").touch()
    (tmp_path / "meta_000003.json").touch()
    (tmp_path / "optim_000003_rank0.pt").touch()
    (tmp_path / "notes.txt").touch()
    assert find_last_step(str(tmp_path)) == 3


def test_raises_file_not_found_when_no_checkpoint_exists(tmp_path):
    (tmp_path / "meta_000003.json").touch()  # a stray sibling file, but no model_*.pt
    with pytest.raises(FileNotFoundError):
        find_last_step(str(tmp_path))


# -- tags: one namespace, the tag is the whole address ------------------------------------------

import os

import torch

from tinylab import checkpoints
from tinylab.checkpoints import resolve_checkpoint_dir, validate_tag


@pytest.mark.parametrize("bad", ["", "/abs", "a//b", "a/", "/a", "..", ".", "../x", "a/../b", "a/./b", "a\\b", "a\0b", None, 3])
def test_validate_tag_rejects_anything_that_could_escape_or_name_nothing(bad):
    with pytest.raises(ValueError):
        validate_tag(bad)


@pytest.mark.parametrize("good", ["d12", "gpt-d12-base", "kvcache/d13-chat", "a/b/c", "2026-09/run.3", "with space", "ünï"])
def test_validate_tag_accepts_arbitrary_text_with_folders(good):
    assert validate_tag(good) == good


def test_the_tag_is_the_path_under_one_checkpoints_dir(base_dir):
    assert resolve_checkpoint_dir("gpt-d12-base") == os.path.join(base_dir, "checkpoints", "gpt-d12-base")
    assert resolve_checkpoint_dir("kvcache/d13-chat") == os.path.join(base_dir, "checkpoints", "kvcache", "d13-chat")
    assert not hasattr(checkpoints, "CHECKPOINT_DIRS")  # the base_/chatsft_ split is gone


def _save_tiny(tag, *, tokenizer_block=None, template="base"):
    """A real (untrained) tiny model saved under `tag` exactly the way ops.train saves one."""
    import dataclasses

    from modelcore import ModelManager
    manager = ModelManager()
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "fixtures", "gpt_tiny.json")) as f:
        import json
        config = manager.config_from_dict(json.load(f))
    config = dataclasses.replace(config, template=template, tokenizer=tokenizer_block)
    model = manager.create_model(config, device=torch.device("cpu"), seed=0)
    checkpoints.save_checkpoint(resolve_checkpoint_dir(tag), 0, model.state_dict(), None,
                                {"step": 0, "model_config": manager.config_to_dict(config)})
    return model


def test_nested_and_flat_tags_coexist_and_load_independently(base_dir):
    _save_tiny("kvcache/d13-chat")
    _save_tiny("kvcache")           # a tag that is also the folder of another
    _save_tiny("gpt-d12-base")
    for tag in ("kvcache/d13-chat", "kvcache", "gpt-d12-base"):
        _, _, meta = checkpoints.load_model(tag, torch.device("cpu"), phase="eval")
        assert meta["model_tag"] == tag
    assert checkpoints.find_last_step(resolve_checkpoint_dir("kvcache")) == 0  # the nested dir is not a checkpoint of it


def test_loading_a_tag_that_does_not_exist_raises(base_dir):
    with pytest.raises(FileNotFoundError):
        checkpoints.load_model("nope/nothing", torch.device("cpu"), phase="eval")


# -- which tokenizer a checkpoint loads with ----------------------------------------------------

def test_a_checkpoint_loads_with_the_tokenizer_its_config_names(base_dir):
    """Selection order: explicit spec > the checkpoint's own recorded name > the default."""
    from tinylab.tokenizer import get_tokenizer
    default = get_tokenizer()
    default.save(os.path.join(base_dir, "tokenizers", "recorded"))
    _save_tiny("m", tokenizer_block=default.descriptor("recorded"))

    _, tok, _ = checkpoints.load_model("m", torch.device("cpu"), phase="eval")           # recorded name, found
    assert tok.fingerprint() == default.fingerprint()

    os.rename(os.path.join(base_dir, "tokenizers", "recorded"), os.path.join(base_dir, "tokenizers", "moved"))
    with pytest.raises(FileNotFoundError, match="recorded"):                               # ...proves it drove the choice
        checkpoints.load_model("m", torch.device("cpu"), phase="eval")
    _, tok, _ = checkpoints.load_model("m", torch.device("cpu"), phase="eval", tokenizer_spec="moved")  # explicit wins
    assert tok.fingerprint() == default.fingerprint()


def test_a_checkpoint_with_no_tokenizer_block_uses_the_default(base_dir):
    """An old (v1-era) checkpoint records no tokenizer -- it loads with the default, as before."""
    _save_tiny("old")
    _, tok, _ = checkpoints.load_model("old", torch.device("cpu"), phase="eval")
    assert tok.get_vocab_size() > 256
