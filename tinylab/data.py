"""
Corpus identity: which URL, which shard count, which HF dataset -- the parts datacore/benchcore
deliberately don't know ("a corpus's URL" is a host concept). Ported and merged from our nanochat fork's
dataset.py + sft_data.py.
"""
import os

from datacore import ExampleSet, load_hub_dataset
from datacore.download import download_shards

from tinylab.runtime import get_base_dir

# -----------------------------------------------------------------------------
# Pretraining corpora: a corpus is a URL template over shard indices, with the last shard held out
# as validation. A job's "corpus" key names one (default: climbmix, karpathy's climbmix-400b-shuffle,
# the upstream nanochat corpus); adding a corpus is one entry here.

CORPORA = {
    "climbmix": {
        "url": "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_{index:05d}.parquet",
        "max_shard": 6542,  # the last data shard, shard_06542.parquet, is the validation shard
        "filename": "shard_{index:05d}.parquet",
    },
}
DEFAULT_CORPUS = "climbmix"


def corpus_spec(corpus):
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
    return CORPORA[corpus]


def corpus_dir(corpus=DEFAULT_CORPUS):
    return os.path.join(get_base_dir(), f"base_data_{corpus}")


def download_corpus_shards(num_train_shards, corpus=DEFAULT_CORPUS, num_workers=4, log=print):
    """Downloads num_train_shards train shards plus the (fixed, last) validation shard of `corpus`,
    skipping any already present. Returns the destination directory."""
    spec = corpus_spec(corpus)
    dest_dir = corpus_dir(corpus)
    os.makedirs(dest_dir, exist_ok=True)
    num_train_shards = min(num_train_shards, spec["max_shard"])
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(spec["max_shard"])  # validation shard is always the last one
    result = download_shards(
        spec["url"], ids_to_download, dest_dir,
        filename_fn=lambda index: spec["filename"].format(index=index), num_workers=num_workers, log=log,
    )
    log(f"Downloaded {result['successful']}/{result['total']} shards to {dest_dir}")
    return dest_dir


def corpus_train_val_paths(num_train_shards, corpus=DEFAULT_CORPUS):
    """Absolute paths for exactly the first `num_train_shards` train shards plus the fixed
    validation shard (always max_shard) -- built from filenames directly, not from however many
    files a previous, larger run happened to leave on disk. A directory-listing-and-slice approach
    would silently pack every shard already present when a later job asks for fewer; this can't,
    because it names the exact files it wants and downloads only those that are missing."""
    spec = corpus_spec(corpus)
    num_train_shards = min(num_train_shards, spec["max_shard"])
    dest_dir = corpus_dir(corpus)
    name = lambda index: os.path.join(dest_dir, spec["filename"].format(index=index))
    return [name(i) for i in range(num_train_shards)], [name(spec["max_shard"])]


# -----------------------------------------------------------------------------
# Chat tasks by name: SmolTalk (training data only) plus benchcore's registry of benchmark tasks,
# matched case-insensitively ("gsm8k" for 'GSM8K'). Used by the sft mixture and the rl task.

def chat_task_name(name):
    """The registry spelling of a chat task name ('gsm8k' -> 'GSM8K'); raises ValueError listing
    the known names. benchcore's own names only -- SmolTalk is handled by build_task."""
    from benchcore import CHAT_TASKS
    by_lower = {registered.lower(): registered for registered in CHAT_TASKS}
    if not isinstance(name, str) or name.lower() not in by_lower:
        raise ValueError(f"unknown task {name!r}; known: {sorted(CHAT_TASKS)} (and 'smoltalk' for sft mixtures)")
    return by_lower[name.lower()]


def build_task(name, split, **task_kwargs):
    """One task for `split` ("train" or "test"), cached under the base dir: SmolTalk, or any task of
    benchcore's registry. `task_kwargs` (e.g. stop=) go to the task's ExampleSet constructor."""
    if isinstance(name, str) and name.lower() == "smoltalk":
        return SmolTalk(split=split, **task_kwargs)
    from benchcore import build_chat_tasks
    registered = chat_task_name(name)
    return build_chat_tasks([registered], cache_dir=get_base_dir(), split=split, **task_kwargs)[registered]


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
