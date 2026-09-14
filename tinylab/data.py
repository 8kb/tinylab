"""
Corpus identity: which URL, which shard count, which HF dataset -- the parts datacore/benchcore
deliberately don't know ("a corpus's URL" is a host concept, see llmllab/docs/subsystem-
conventions.md). Ported and merged from nanochat's nanochat/dataset.py + nanochat/sft_data.py
(llmllab/nanochat).
"""
import os

from datacore import ExampleSet, load_hub_dataset
from datacore.download import download_shards

from tinylab.runtime import get_base_dir

# -----------------------------------------------------------------------------
# ClimbMix: the pretraining corpus. Same source nanochat uses, so a shard downloaded once could in
# principle be reused -- but tinylab keeps its own base dir (see AGENTS.md), so it re-downloads.

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # the last data shard is shard_06542.parquet
_index_to_filename = lambda index: f"shard_{index:05d}.parquet"


def climbmix_dir():
    return os.path.join(get_base_dir(), "base_data_climbmix")


def download_climbmix_shards(num_train_shards, num_workers=4, log=print):
    """Downloads num_train_shards train shards plus the (fixed, last) validation shard, skipping
    any already present. Returns the destination directory."""
    dest_dir = climbmix_dir()
    os.makedirs(dest_dir, exist_ok=True)
    num_train_shards = min(num_train_shards, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)  # validation shard is always the last one
    result = download_shards(
        BASE_URL + "/shard_{index:05d}.parquet", ids_to_download, dest_dir,
        filename_fn=_index_to_filename, num_workers=num_workers, log=log,
    )
    log(f"Downloaded {result['successful']}/{result['total']} shards to {dest_dir}")
    return dest_dir


def climbmix_train_val_paths(num_train_shards):
    """Absolute paths for exactly the first `num_train_shards` train shards plus the fixed
    validation shard (always MAX_SHARD) -- built from filenames directly, not from however many
    files a previous, larger run happened to leave on disk. A directory-listing-and-slice approach
    would silently pack every shard already present when a later job asks for fewer; this can't,
    because it names the exact files it wants and downloads only those that are missing."""
    num_train_shards = min(num_train_shards, MAX_SHARD)
    dest_dir = climbmix_dir()
    train_paths = [os.path.join(dest_dir, _index_to_filename(i)) for i in range(num_train_shards)]
    val_paths = [os.path.join(dest_dir, _index_to_filename(MAX_SHARD))]
    return train_paths, val_paths


# -----------------------------------------------------------------------------
# SmolTalk: general-purpose SFT conversation data. Pure training data with no eval criterion, so
# it's built directly on datacore.ExampleSet + load_hub_dataset rather than benchcore.Task (a new
# benchmark task belongs in benchcore; a new training-data container belongs here).


class SmolTalk(ExampleSet):
    """smol-smoltalk (HuggingFaceTB): a general-purpose conversational SFT dataset, sized for
    smaller models -- see https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk."""

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "test"), "SmolTalk split must be train|test"
        self.ds = load_hub_dataset("HuggingFaceTB/smol-smoltalk", split=split, cache_dir=get_base_dir()).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        assert len(messages) >= 1
        rest_messages = messages[1:] if messages[0]["role"] == "system" else messages
        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        return {"messages": messages}
