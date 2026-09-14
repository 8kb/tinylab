"""
The `bench` op: score a checkpoint against benchcore's CORE suite or its chat-task suite (ARC/
MMLU/GSM8K/HumanEval). Ported and trimmed from nanochat's scripts/base_eval.py + scripts/
chat_eval.py (llmllab/nanochat) -- unified into one op selected by "suite" instead of two scripts.

A modelcore Model already satisfies benchcore.Model (it's __call__(input_ids) -> logits and has
get_device()); tinylab.engine.Engine is the Generator adapter benchcore's generative tasks
(GSM8K, HumanEval) need -- see benchcore/AGENTS.md's protocols.py for why that adapter has to live
in the host, not in either subsystem.
"""
from tinylab import checkpoints
from tinylab.engine import Engine
from tinylab.runtime import get_base_dir, print0

# Keys shared by both suites, plus each suite's own -- see accepted_keys() below, which is
# suite-aware so e.g. "max_per_task" (a CORE-only concept) on a suite="chat" step is caught as an
# error rather than silently accepted and ignored.
_COMMON_KEYS = {"suite", "source", "model_tag", "model_step"}
_CORE_KEYS = {"max_per_task"}
_CHAT_KEYS = {"tasks", "batch_size", "num_samples", "max_new_tokens", "temperature", "top_k", "max_problems"}


def accepted_keys(cfg: dict) -> set:
    suite = cfg.get("suite")
    suite_keys = _CORE_KEYS if suite == "core" else _CHAT_KEYS if suite == "chat" else (_CORE_KEYS | _CHAT_KEYS)
    return _COMMON_KEYS | suite_keys


def _load(cfg, ctx):
    assert "model_tag" in cfg, "bench: 'model_tag' is required"
    source = cfg.get("source", "sft")
    return checkpoints.load_model(source, ctx.device, phase="eval", model_tag=cfg["model_tag"], step=cfg.get("model_step"))


def _run_core(cfg, ctx, model, tokenizer):
    from benchcore import load_core_suite
    suite = load_core_suite(get_base_dir(), max_per_task=cfg.get("max_per_task"))
    report = ctx.bench_manager.core(model, tokenizer, suite, device=ctx.device, rank=ctx.rank, world_size=ctx.world_size)
    for label, acc in report.results.items():
        print0(f"  {label:30s} acc={acc:.4f}  centered={report.centered_results[label]:.4f}")
    print0(f"CORE metric: {report.core_metric:.4f}")
    return {"op": "bench", "suite": "core", "core_metric": report.core_metric, "results": report.results}


def _run_chat(cfg, ctx, model, tokenizer):
    from benchcore import ARC, ALL_CHAT_TASKS, GSM8K, HumanEval, MMLU
    cache_dir = get_base_dir()
    builders = {
        "ARC-Easy": lambda: ARC(subset="ARC-Easy", split="test", cache_dir=cache_dir),
        "ARC-Challenge": lambda: ARC(subset="ARC-Challenge", split="test", cache_dir=cache_dir),
        "MMLU": lambda: MMLU(subset="all", split="test", cache_dir=cache_dir),
        "GSM8K": lambda: GSM8K(subset="main", split="test", cache_dir=cache_dir),
        "HumanEval": lambda: HumanEval(cache_dir=cache_dir),
    }
    task_names = cfg.get("tasks", list(ALL_CHAT_TASKS))
    unknown = set(task_names) - set(builders)
    assert not unknown, f"bench: unknown chat task(s) {sorted(unknown)} (available: {sorted(builders)})"
    tasks = {name: builders[name]() for name in task_names}
    generator = Engine(model, tokenizer, manager=ctx.model_manager)
    report = ctx.bench_manager.chat_suite(
        tasks, model, tokenizer, generator=generator, batch_size=cfg.get("batch_size", 1),
        num_samples=cfg.get("num_samples", 1), max_new_tokens=cfg.get("max_new_tokens", 256),
        temperature=cfg.get("temperature", 0.0), top_k=cfg.get("top_k", 50), max_problems=cfg.get("max_problems"),
        device=ctx.device, rank=ctx.rank, world_size=ctx.world_size,
    )
    for name, acc in report.results.items():
        print0(f"  {name:15s} acc={acc:.4f}")
    if report.chatcore_metric is not None:
        print0(f"ChatCORE metric: {report.chatcore_metric:.4f}")
    return {"op": "bench", "suite": "chat", "chatcore_metric": report.chatcore_metric, "results": report.results}


def run(cfg: dict, ctx) -> dict:
    assert "suite" in cfg, "bench: 'suite' is required ('core' or 'chat')"
    suite = cfg["suite"]
    assert suite in ("core", "chat"), f"bench: suite must be 'core' or 'chat', got {suite!r}"
    model, tokenizer, _meta = _load(cfg, ctx)
    model.eval()
    if suite == "core":
        return _run_core(cfg, ctx, model, tokenizer)
    return _run_chat(cfg, ctx, model, tokenizer)
