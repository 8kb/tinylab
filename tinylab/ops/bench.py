"""
The `bench` op: score a checkpoint against benchcore's CORE suite or its chat-task suite (ARC/
MMLU/GSM8K/HumanEval). Ported from nanochat's scripts/base_eval.py + scripts/chat_eval.py, unified
into one op selected by "suite" instead of two scripts -- see docs/architecture.md for the
Model/Generator adapter split, docs/job-file.md for every cfg key this module reads.
"""
from tinylab import checkpoints
from tinylab.engine import Engine
from tinylab.runtime import get_base_dir, print0

# Keys shared by both suites, plus each suite's own -- see accepted_keys() below, which is
# suite-aware so e.g. "max_per_task" (a CORE-only concept) on a suite="chat" step is caught as an
# error rather than silently accepted and ignored.
_COMMON_KEYS = {"suite", "model_tag", "model_step"}
_CORE_KEYS = {"max_per_task"}
_CHAT_KEYS = {"tasks", "batch_size", "num_samples", "max_new_tokens", "temperature", "top_k", "max_problems",
              "generative_batch_size", "eval_workers"}


def accepted_keys(cfg: dict) -> set:
    suite = cfg.get("suite")
    suite_keys = _CORE_KEYS if suite == "core" else _CHAT_KEYS if suite == "chat" else (_CORE_KEYS | _CHAT_KEYS)
    return _COMMON_KEYS | suite_keys


def _load(cfg, ctx):
    """Loads the checkpoint cfg names: cfg["model_tag"] is its whole address, cfg["model_step"] a
    specific step (default: latest). tokenizer_spec is this step's own "tokenizer" key if set
    (see Context.tokenizer_for), else None -- checkpoints.load_model then falls back to whatever
    tokenizer the checkpoint's own config says it was trained with. Returns (model, tokenizer,
    meta_data)."""
    assert "model_tag" in cfg, "bench: 'model_tag' is required"
    return checkpoints.load_model(cfg["model_tag"], ctx.device, phase="eval", step=cfg.get("model_step"),
                                  tokenizer_spec=cfg.get("tokenizer"))


def _run_core(cfg, ctx, model, tokenizer):
    """suite="core": scores model against DCLM's CORE suite, capped at cfg["max_per_task"]
    examples per task if given (must leave enough for each task's own few-shot count -- see
    AGENTS.md). Returns {"op": "bench", "suite": "core", "core_metric", "results"}."""
    report = ctx.bench_manager.core_suite(model, tokenizer, cache_dir=get_base_dir(), max_per_task=cfg.get("max_per_task"),
                                           device=ctx.device, rank=ctx.rank, world_size=ctx.world_size)
    for label, acc in report.results.items():
        print0(f"  {label:30s} acc={acc:.4f}  centered={report.centered_results[label]:.4f}")
    print0(f"CORE metric: {report.core_metric:.4f}")
    return {"op": "bench", "suite": "core", "core_metric": report.core_metric, "results": report.results}


def _run_chat(cfg, ctx, model, tokenizer):
    """suite="chat": scores model against cfg["tasks"] (default: all of ARC-Easy, ARC-Challenge,
    MMLU, GSM8K, HumanEval), generating with tinylab.engine.Engine as the benchcore.Generator
    adapter. "generative_batch_size" (default 1) and "eval_workers" (default 1) are deliberately
    separate from "batch_size": batch_size stays "problems per forward" in the categorical
    (ARC/MMLU) loop, and reusing it for the generative (GSM8K/HumanEval) loop would silently
    change the numbers of any existing job that already sets batch_size. Returns
    {"op": "bench", "suite": "chat", "chatcore_metric", "results"}."""
    from benchcore import build_chat_tasks
    task_names = cfg.get("tasks")
    try:
        tasks = build_chat_tasks(task_names, cache_dir=get_base_dir())
    except ValueError as e:
        raise AssertionError(f"bench: {e}")
    generator = Engine(model, tokenizer, manager=ctx.model_manager)
    report = ctx.bench_manager.chat_suite(
        tasks, model, tokenizer, generator=generator, batch_size=cfg.get("batch_size", 1),
        num_samples=cfg.get("num_samples", 1), max_new_tokens=cfg.get("max_new_tokens", 256),
        temperature=cfg.get("temperature", 0.0), top_k=cfg.get("top_k", 50), max_problems=cfg.get("max_problems"),
        generative_batch_size=cfg.get("generative_batch_size", 1), eval_workers=cfg.get("eval_workers", 1),
        device=ctx.device, rank=ctx.rank, world_size=ctx.world_size,
    )
    for name, acc in report.results.items():
        print0(f"  {name:15s} acc={acc:.4f}")
    if report.chatcore_metric is not None:
        print0(f"ChatCORE metric: {report.chatcore_metric:.4f}")
    return {"op": "bench", "suite": "chat", "chatcore_metric": report.chatcore_metric, "results": report.results}


def run(cfg: dict, ctx) -> dict:
    """Runs one bench step: cfg is a resolved job-file step (see docs/job-file.md for every key),
    ctx the shared Context for this job run."""
    assert "suite" in cfg, "bench: 'suite' is required ('core' or 'chat')"
    suite = cfg["suite"]
    assert suite in ("core", "chat"), f"bench: suite must be 'core' or 'chat', got {suite!r}"
    model, tokenizer, _meta = _load(cfg, ctx)
    model.eval()
    if suite == "core":
        return _run_core(cfg, ctx, model, tokenizer)
    return _run_chat(cfg, ctx, model, tokenizer)
