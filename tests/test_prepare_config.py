"""prepare's configurability: the corpus key and the SFT mixture, resolved without any download."""
import pytest

from tinylab import data
from tinylab.ops import prepare


def test_the_default_mixture_is_the_historical_one():
    assert prepare.resolve_mixture({}) == [
        {"task": "smoltalk", "epochs": 1},
        {"task": "mmlu", "epochs": 3, "val_cap": 5200},
        {"task": "gsm8k", "epochs": 4, "val_cap": 420},
    ]


def test_the_older_epoch_keys_still_work_as_aliases():
    mixture = prepare.resolve_mixture({"mmlu_epochs": 1, "gsm8k_epochs": 2})
    assert [(e["task"], e["epochs"]) for e in mixture] == [("smoltalk", 1), ("mmlu", 1), ("gsm8k", 2)]


def test_giving_both_forms_is_an_error():
    with pytest.raises(AssertionError, match="replaces"):
        prepare.resolve_mixture({"mixture": [{"task": "smoltalk"}], "mmlu_epochs": 2})


@pytest.mark.parametrize("bad, match", [
    ([], "non-empty"),
    ([{"epochs": 2}], 'needs a "task"'),
    ([{"task": "smoltalk", "weight": 2}], "unknown key"),
    ([{"task": "smoltalk", "epochs": 0}], "epochs must be >= 1"),
])
def test_bad_mixtures_are_rejected(bad, match):
    with pytest.raises(AssertionError, match=match):
        prepare.resolve_mixture({"mixture": bad})


def test_an_unknown_task_names_the_known_ones():
    with pytest.raises(ValueError, match="known:"):
        prepare.resolve_mixture({"mixture": [{"task": "nope"}]})


def test_the_mixture_repeats_train_splits_by_epochs_and_caps_val(monkeypatch):
    built = []
    monkeypatch.setattr(data, "build_task", lambda name, split, **kw: built.append((name, split, kw)) or _Fixed(3))
    train, val = prepare._build_sft_mixtures({"mixture": [{"task": "smoltalk", "epochs": 2}, {"task": "ARC-Easy", "val_cap": 1}]}, None)
    assert [b for b in built if b[1] == "train"] == [("smoltalk", "train", {})] * 2 + [("ARC-Easy", "train", {})]
    assert ("ARC-Easy", "test", {"stop": 1}) in built and ("smoltalk", "test", {}) in built
    assert len(train) == 9 and len(val) == 6


class _Fixed:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i

    start, stop, step = 0, None, 1

    def num_examples(self):
        return self.n


def test_corpora_resolve_paths_and_the_default_name_follows_the_corpus(base_dir, monkeypatch):
    monkeypatch.setitem(data.CORPORA, "tiny", {"url": "http://x/{index}.parquet", "max_shard": 3, "filename": "s{index}.parquet"})
    train, val = data.corpus_train_val_paths(2, "tiny")
    assert [p.rsplit("/", 1)[1] for p in train] == ["s0.parquet", "s1.parquet"] and val[0].endswith("s3.parquet")
    assert "base_data_tiny" in train[0]

    class Tok:
        def fingerprint(self):
            return "fp"
    assert prepare.default_dataset_name("base", 64, Tok(), "tiny") == "tiny_t64_fp"
    assert prepare.default_dataset_name("base", 64, Tok()) == "climbmix_t64_fp"
    with pytest.raises(ValueError, match="unknown corpus"):
        data.corpus_spec("nope")
