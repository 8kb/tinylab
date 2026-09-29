"""
The strongest bucket test: train against a MemoryRemote, wipe the local disk, and resume from the
bucket alone -- model, optimizer and dataloader cursor all make the round trip, and the dataset
streams back in. The resumed run must land on exactly the val_bpb a continuous run does.

Marked slow (real tiny training runs), like test_resume.py.
"""
import os
import shutil

import pytest
from datacore import BestFitCropPacker, DataManager

from tinylab import remote as R
from tinylab.ops import Context
from tinylab.ops.train import run as train_run
from tinylab.remote_stores import UploadingDatasetStore
from tinylab.tokenizer import get_tokenizer

from test_resume import _SENTENCES, _FakeTextSource, _base_cfg

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def fast_uploader(monkeypatch):
    monkeypatch.setattr(R.Uploader, "IDLE_SECONDS", 0.05)


def _ctx(remote, *, resume, uploader=True):
    return Context(device_type="cpu", resume=resume, remote=remote, uploader=R.Uploader(remote) if uploader else None,
                   experiment="scratch", job_name="rt")


def test_resume_from_the_bucket_on_a_clean_base_dir_matches_a_continuous_run(base_dir):
    sequence_len = 32
    tokenizer = get_tokenizer(base_dir)
    remote = R.MemoryRemote("roundtrip")

    # prepare -> the dataset goes to the bucket as it is written
    uploader = R.Uploader(remote)
    store = UploadingDatasetStore(os.path.join(base_dir, "prepared", "smoke"), name="smoke", uploader=uploader,
                                  producer={"repo": "tinylab", "experiment": "scratch", "job": "rt", "step": "data"})
    DataManager().prepare(store, sources={"train": _FakeTextSource(_SENTENCES, repeats=50), "val": _FakeTextSource(_SENTENCES, repeats=5)},
                          tokenizer=tokenizer, sequence_len=sequence_len, sequences_per_volume=64,
                          packer=BestFitCropPacker(buffer_size=64))
    uploader.flush()

    continuous = train_run(dict(_base_cfg(sequence_len), num_iterations=6), Context(device_type="cpu"))
    shutil.rmtree(os.path.join(base_dir, "checkpoints"))

    ctx = _ctx(remote, resume=False)
    interrupted = train_run(dict(_base_cfg(sequence_len), num_iterations=3), ctx)
    ctx.uploader.flush()
    assert interrupted["step"] == 3
    # "last" retention: exactly one resumable state is in the bucket
    entity = "checkpoints/pre"
    assert R.complete_steps(remote.list(entity), entity) == [3]
    assert "checkpoints/pre/optim_000003_rank0.pt" in remote.files

    # a fresh pod: nothing local but the (bundled) default tokenizer
    shutil.rmtree(os.path.join(base_dir, "checkpoints"))
    shutil.rmtree(os.path.join(base_dir, "prepared"))
    resumed = train_run(dict(_base_cfg(sequence_len), num_iterations=6), _ctx(remote, resume=True, uploader=False))

    assert resumed["step"] == 6
    assert resumed["val_bpb"] == pytest.approx(continuous["val_bpb"])
    assert os.path.exists(os.path.join(base_dir, "prepared", "smoke", "manifest.json"))  # streamed back from the bucket


def test_sft_style_read_of_a_checkpoint_pulls_only_model_and_meta(base_dir):
    from tinylab import checkpoints
    sequence_len = 32
    tokenizer = get_tokenizer(base_dir)
    remote = R.MemoryRemote("pullonly")
    from test_resume import _prepare_fake_dataset
    _prepare_fake_dataset(base_dir, tokenizer, sequence_len)
    ctx = _ctx(remote, resume=False)
    train_run(dict(_base_cfg(sequence_len), num_iterations=3), ctx)
    ctx.uploader.flush()
    shutil.rmtree(os.path.join(base_dir, "checkpoints"))

    model, _tok, meta = checkpoints.load_model("pre", "cpu", phase="eval", remote=remote)
    assert meta["step"] == 3
    assert sorted(os.listdir(os.path.join(base_dir, "checkpoints", "pre"))) == ["meta_000003.json", "model_000003.pt"]
