"""
The op registry and the Context every op runs with. An op module exposes two things:
`accepted_keys(cfg: dict) -> set[str]` (the job-file keys tinylab.job validates a step against --
a function, not a static set, so an op like `prepare`/`train` can accept a different key set
depending on the step's own "kind") and `run(cfg: dict, ctx: Context) -> dict`.

Context carries base dir, device, and rank/world_size, plus lazily-built ModelManager/DataManager/
BenchManager/tokenizer singletons -- explicitly passed to every op, never a module global, matching
the subsystems' own no-ambient-globals rule (see llmllab/docs/subsystem-conventions.md).
"""
from dataclasses import dataclass, field


@dataclass
class Context:
    device_type: str = "auto"
    _device_info: tuple | None = field(default=None, repr=False, compare=False)
    _model_manager: object | None = field(default=None, repr=False, compare=False)
    _data_manager: object | None = field(default=None, repr=False, compare=False)
    _bench_manager: object | None = field(default=None, repr=False, compare=False)
    _tokenizer: object | None = field(default=None, repr=False, compare=False)

    @property
    def device_info(self):
        """(ddp, ddp_rank, ddp_local_rank, ddp_world_size, device), computed once and reused by
        every step in a run so a multi-step job doesn't re-init the process group."""
        if self._device_info is None:
            from tinylab.runtime import compute_init
            self._device_info = compute_init(self.device_type)
        return self._device_info

    @property
    def device(self):
        return self.device_info[4]

    @property
    def rank(self):
        return self.device_info[1]

    @property
    def world_size(self):
        return self.device_info[3]

    @property
    def model_manager(self):
        if self._model_manager is None:
            from modelcore import ModelManager
            self._model_manager = ModelManager()
        return self._model_manager

    @property
    def data_manager(self):
        if self._data_manager is None:
            from datacore import DataManager
            self._data_manager = DataManager()
        return self._data_manager

    @property
    def bench_manager(self):
        if self._bench_manager is None:
            from benchcore import BenchManager
            self._bench_manager = BenchManager()
        return self._bench_manager

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from tinylab.tokenizer import get_tokenizer
            self._tokenizer = get_tokenizer()
        return self._tokenizer


# Keys every op accepts even if it doesn't read all of them, because they're meant to live in a
# job file's shared "defaults" block: "device" (compute_init's device_type), "sequence_len" (must
# agree between a prepared dataset and the model that trains on it), "model_config" (a path to a
# materialized ModelConfig tree -- irrelevant to prepare/bench, which is fine, they just ignore
# it), "world_size" (the GPU count a train step's total_batch_size/grad_accum math assumes --
# checked against the actual launch, not read by prepare/bench, but a defaults-block value shared
# by every step in a run either way).
COMMON_KEYS = {"device", "sequence_len", "model_config", "world_size"}


from tinylab.ops import prepare, train, bench  # noqa: E402 -- after Context, to avoid a cycle

OPS = {
    "prepare": prepare,
    "train": train,
    "bench": bench,
}
