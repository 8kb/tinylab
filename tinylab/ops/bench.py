"""
The `bench` op: measure a checkpoint (or a tokenizer) with one of six suites, selected by "suite":
"core" (DCLM CORE), "chat" (ARC/MMLU/GSM8K/HumanEval via benchcore), "bpb" (bits per byte on a
prepared dataset's train and val splits), "sample" (fixed-prompt completions), "infer" (CUDA
latency/throughput/MBU sweep) and "tokenizer" (compression ratio; needs no model). Ported from
our nanochat fork's scripts/base_eval.py, chat_eval.py, infer_bench.py and tok_eval.py (see
llmllab/docs/history.md) -- see docs/architecture.md for the Model/Generator adapter split,
docs/job-file.md for every cfg key this module reads.
"""
import json
import os
import time

from tinylab import bench_texts, checkpoints
from tinylab import data as data_mod
from tinylab.engine import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TOP_K, Engine
from tinylab.runtime import get_base_dir, print0

# Keys shared by every suite, plus each suite's own -- see accepted_keys() below, which is
# suite-aware so e.g. "max_per_task" (a CORE-only concept) on a suite="chat" step is caught as an
# error rather than silently accepted and ignored. suite="tokenizer" loads no checkpoint, so its
# "model_tag"/"model_step" are not accepted (it measures the step's own "tokenizer").
_COMMON_KEYS = {"suite", "model_tag", "model_step"}
_SUITE_KEYS = {
    "core": {"max_per_task"},
    "chat": {"tasks", "batch_size", "num_samples", "max_new_tokens", "temperature", "top_k", "max_problems",
             "generative_batch_size", "eval_workers"},
    "bpb": {"dataset", "corpus", "split_tokens", "device_batch_size"},
    "sample": {"prompts", "max_new_tokens", "temperature", "top_k"},
    "infer": {"prompt_tokens", "decode_tokens", "batch_sizes", "temperature"},
    "tokenizer": {"baselines", "corpus"},
}
SUITES = tuple(_SUITE_KEYS)
_NO_MODEL_SUITES = {"tokenizer"}


def accepted_keys(cfg: dict) -> set:
    suite = cfg.get("suite")
    if suite in _SUITE_KEYS:
        keys = _SUITE_KEYS[suite]
        return {"suite"} | keys if suite in _NO_MODEL_SUITES else _COMMON_KEYS | keys
    return _COMMON_KEYS.union(*_SUITE_KEYS.values())


def _load(cfg, ctx):
    """Loads the checkpoint cfg names: cfg["model_tag"] is its whole address, cfg["model_step"] a
    specific step (default: latest). tokenizer_spec is this step's own "tokenizer" key if set
    (see Context.tokenizer_for), else None -- checkpoints.load_model then falls back to whatever
    tokenizer the checkpoint's own config says it was trained with. Returns (model, tokenizer,
    meta_data)."""
    assert "model_tag" in cfg, "bench: 'model_tag' is required"
    return checkpoints.load_model(cfg["model_tag"], ctx.device, phase="eval", step=cfg.get("model_step"),
                                  tokenizer_spec=cfg.get("tokenizer"), remote=ctx.remote)


def _run_core(cfg, ctx, model, tokenizer):
    """suite="core": scores model against DCLM's CORE suite, capped at cfg["max_per_task"]
    examples per task if given (must leave enough for each task's own few-shot count -- see
    AGENTS.md). Returns {"op": "bench", "suite": "core", "core_metric", "results"}."""
    if ctx.remote is not None:
        from tinylab import remote as remote_mod
        remote_mod.pull_prefix(ctx.remote, "eval_bundle")  # only files not already local
    report = ctx.bench_manager.core_suite(model, tokenizer, cache_dir=get_base_dir(), max_per_task=cfg.get("max_per_task"),
                                           device=ctx.device, rank=ctx.rank, world_size=ctx.world_size, log=print0)
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
    if ctx.remote is not None:
        from tinylab import remote as remote_mod
        remote_mod.pull_prefix(ctx.remote, remote_mod.TASK_DATA)  # bench sets only; SmolTalk is never in the bucket
    task_names = cfg.get("tasks")
    try:
        tasks = build_chat_tasks(task_names, cache_dir=get_base_dir())
    except ValueError as e:
        raise AssertionError(f"bench: {e}")
    generator = Engine(model, tokenizer, manager=ctx.model_manager)
    report = ctx.bench_manager.chat_suite(
        tasks, model, tokenizer, generator=generator, batch_size=cfg.get("batch_size", 1),
        num_samples=cfg.get("num_samples", 1), max_new_tokens=cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
        temperature=cfg.get("temperature", 0.0), top_k=cfg.get("top_k", DEFAULT_TOP_K), max_problems=cfg.get("max_problems"),
        generative_batch_size=cfg.get("generative_batch_size", 1), eval_workers=cfg.get("eval_workers", 1),
        device=ctx.device, rank=ctx.rank, world_size=ctx.world_size, log=print0,
    )
    for name, acc in report.results.items():
        print0(f"  {name:15s} acc={acc:.4f}")
    if report.chatcore_metric is not None:
        print0(f"ChatCORE metric: {report.chatcore_metric:.4f}")
    return {"op": "bench", "suite": "chat", "chatcore_metric": report.chatcore_metric, "results": report.results}


def _run_bpb(cfg, ctx, model, tokenizer):
    """suite="bpb": bits per byte of model on the train and val splits of a prepared dataset
    (cfg["dataset"], else the default name for this model's sequence_len and the tokenizer),
    split_tokens (default 20,971,520) tokens per split, rounded down to a whole number of eval
    batches. Returns {"op": "bench", "suite": "bpb", "results": {"train", "val"}}."""
    from tinylab.ops.prepare import open_prepared
    sequence_len = model.config.sequence_len
    device_batch_size = cfg.get("device_batch_size", 32)
    tokens_per_step = device_batch_size * sequence_len * ctx.world_size
    split_tokens = cfg.get("split_tokens", 40 * 524288)
    steps = max(1, split_tokens // tokens_per_step)
    _name, dataset, token_bytes = open_prepared(cfg, ctx, "base", sequence_len, tokenizer)
    results = {}
    for split in ("train", "val"):
        loader = ctx.data_manager.batches(dataset, split, device_batch_size, device=ctx.device, rank=ctx.rank,
                                          world_size=ctx.world_size, infinite=True)
        results[split] = ctx.model_manager.evaluate_bpb(model, loader, steps, token_bytes)
        print0(f"  {split} bpb: {results[split]:.6f}  ({steps * tokens_per_step:,} tokens)")
    return {"op": "bench", "suite": "bpb", "results": results}


def _run_sample(cfg, ctx, model, tokenizer):
    """suite="sample": for each prompt (cfg["prompts"], default bench_texts.SAMPLE_PROMPTS), one
    greedy completion and one sampled at cfg["temperature"] (default 1.0), up to cfg["max_new_tokens"]
    (default 32) tokens each. Not a score -- read the text. Returns {"op": "bench", "suite":
    "sample", "samples": [{"prompt", "greedy", "sampled"}]}."""
    prompts = cfg.get("prompts", bench_texts.SAMPLE_PROMPTS)
    max_new_tokens = cfg.get("max_new_tokens", 32)
    temperature = cfg.get("temperature", 1.0)
    top_k = cfg.get("top_k", DEFAULT_TOP_K)
    engine = Engine(model, tokenizer, manager=ctx.model_manager)
    samples = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, prepend=tokenizer.ids.bos)
        greedy, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=max_new_tokens, temperature=0.0, top_k=top_k)
        sampled, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        row = {"prompt": prompt, "greedy": tokenizer.decode(greedy[0][len(tokens):]),
               "sampled": tokenizer.decode(sampled[0][len(tokens):])}
        samples.append(row)
        print0("-" * 80)
        print0(f"{prompt!r}\n  greedy:  {row['greedy']!r}\n  sampled: {row['sampled']!r}")
    return {"op": "bench", "suite": "sample", "samples": samples}


def _build_prompt(tokenizer, num_tokens):
    """A natural-language prompt of exactly num_tokens tokens (random ids would do for speed, but
    a real prompt keeps argmax decoding from degenerating)."""
    paragraph = ("The history of science is the study of the development of science, "
                 "including both the natural and social sciences. Science is a body of "
                 "empirical, theoretical, and practical knowledge about the natural world. ")
    tokens = tokenizer.encode(paragraph * (num_tokens // 10), prepend=tokenizer.ids.bos)
    assert len(tokens) >= num_tokens, "prompt text too short, increase the repetition"
    return tokens[:num_tokens]


def _bench_generate(engine, prompt_tokens, batch_size, decode_tokens, temperature):
    """One timed generation: TTFT (first next(): batch-1 prefill, KV replication to batch_size
    rows, first sample), then every later next() is one decode step for all rows."""
    import torch
    device = engine.model.get_device()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    generator = engine.generate(prompt_tokens, num_samples=batch_size, max_tokens=decode_tokens, temperature=temperature)
    t_start = time.perf_counter()
    next(generator)
    torch.cuda.synchronize(device)
    ttft = time.perf_counter() - t_start
    step_times = []
    while True:
        t0 = time.perf_counter()
        try:
            next(generator)
        except StopIteration:
            break
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)
    return dict(ttft=ttft, step_times=step_times, peak_vram=torch.cuda.max_memory_allocated(device))


def _run_infer(cfg, ctx, model, tokenizer, meta):
    """suite="infer": latency, throughput, memory and bandwidth utilization of model, sweeping the
    decode batch size. Prefill (compute-bound) is judged by MFU, decode (every step re-reads all
    weights and the KV cache: bandwidth-bound) by MBU -- achieved bytes/sec over the GPU's peak
    (modelcore.runtime.peak_bandwidth). Single GPU, CUDA only. Returns {"op": "bench", "suite":
    "infer", ...the full static card and the sweep}, all JSON-serializable (unknown GPU peaks are
    None, not Infinity)."""
    import torch
    from modelcore import runtime as modelcore_runtime
    manager = ctx.model_manager
    config = model.config
    stats = manager.stats(config)
    shape = stats.shape_summary
    engine = Engine(model, tokenizer, manager=manager)
    decode_tokens = cfg.get("decode_tokens", 256)
    temperature = cfg.get("temperature", 0.0)
    room = config.sequence_len - decode_tokens  # prompt + decode must fit
    prompt_len = min(cfg.get("prompt_tokens", room), room)
    assert prompt_len > 0, f"bench infer: decode_tokens={decode_tokens} leaves no room in sequence_len={config.sequence_len}"
    prompt_tokens = _build_prompt(tokenizer, prompt_len)

    device = ctx.device
    device_name = torch.cuda.get_device_name(device)
    peak_bw = modelcore_runtime.peak_bandwidth(device_name, log=print0)
    peak_flops = modelcore_runtime.peak_flops(device_name, log=print0)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters())
    dtype_counts = {}
    for p in model.parameters():
        name = str(p.dtype).replace("torch.", "")
        dtype_counts[name] = dtype_counts.get(name, 0) + p.numel()
    kv_store = stats.kv_bytes_per_token()
    context_mid = prompt_len + decode_tokens // 2  # representative decode context
    kv_read = stats.kv_read_bytes(context_mid)
    ceiling_bs1 = peak_bw / (w_bytes + kv_read)  # tok/s at batch 1: every step re-reads weights + KV
    max_rows = int((total_vram - w_bytes) / (kv_store * config.sequence_len))  # full-context rows next to the weights
    inf_to_none = lambda v: None if v == float("inf") else v

    print0(f"Model: {cfg['model_tag']} (step {meta['step']}) | depth {shape['n_layer']}, dim {shape['n_embd']}, "
           f"heads {shape['n_head']}, kv heads {shape['n_kv_head']}")
    print0(f"GPU: {device_name} | peak bandwidth {peak_bw / 1e12:.2f} TB/s | peak compute {peak_flops / 1e12:.0f} TFLOPS | VRAM {total_vram / 2**30:.0f} GiB")
    print0(f"Parameters: {num_params:,} | weight bytes as stored: {w_bytes / 2**20:.0f} MiB | KV cache: {kv_store:,} bytes/token stored, "
           f"{kv_read:,} read/step at context {context_mid}")
    print0(f"Theoretical decode ceiling at batch 1: {ceiling_bs1:,.0f} tok/s | max ~{max_rows:,} full-context rows in VRAM")

    payload = {
        "model_tag": cfg["model_tag"], "step": meta["step"], "gpu": device_name,
        "peak_bandwidth_bytes_per_sec": inf_to_none(peak_bw), "peak_flops_per_sec": inf_to_none(peak_flops),
        "total_vram_bytes": total_vram, "num_params": num_params, "param_dtypes": dtype_counts, "weight_bytes": w_bytes,
        "kv_bytes_per_token": kv_store, "kv_read_bytes_per_step": kv_read, "context_mid": context_mid,
        "decode_flops_per_token": stats.decode_flops(context_mid),
        "ceiling_bs1_tok_per_sec": round(ceiling_bs1, 1) if ceiling_bs1 != float("inf") else None,
        "max_full_context_rows": max_rows, "prompt_tokens": prompt_len, "decode_tokens": decode_tokens,
        "temperature": temperature, "sweep": [],
    }

    # Prefill: batch 1, two decode steps, so TTFT ~= prefill time.
    _bench_generate(engine, prompt_tokens, 1, 2, temperature)  # warmup
    prefill_time = _bench_generate(engine, prompt_tokens, 1, 2, temperature)["ttft"]
    prefill_mfu = 100 * stats.prefill_flops(prompt_len) / prefill_time / peak_flops
    print0(f"Prefill (batch 1, {prompt_len} tokens): {prompt_len / prefill_time:,.0f} tok/s | MFU {prefill_mfu:.1f}%")
    payload["prefill"] = {"tok_per_sec": round(prompt_len / prefill_time, 1), "mfu_percent": round(prefill_mfu, 2),
                          "time_sec": round(prefill_time, 6)}

    header = f"{'batch':>6} {'TTFT ms':>9} {'TPOT ms':>9} {'tok/s':>10} {'MBU %':>7} {'MFU %':>7} {'VRAM GiB':>9} {'steps':>6}"
    print0(header)
    for batch_size in cfg.get("batch_sizes", [1, 8, 32, 128]):
        _bench_generate(engine, prompt_tokens, batch_size, 8, temperature)  # warmup (autotune, allocator, kernels)
        result = _bench_generate(engine, prompt_tokens, batch_size, decode_tokens, temperature)
        step_times = result["step_times"]
        if not step_times:
            print0(f"{batch_size:>6}  all rows terminated during warmup, skipping")
            continue
        tpot = sorted(step_times)[len(step_times) // 2]  # median decode step time
        tok_per_sec = batch_size * len(step_times) / sum(step_times)
        mbu = 100 * ((w_bytes + batch_size * kv_read) / tpot) / peak_bw  # distance from the bandwidth roofline
        mfu = 100 * (batch_size * stats.decode_flops(context_mid) / tpot) / peak_flops  # ... and the compute one
        print0(f"{batch_size:>6} {result['ttft'] * 1e3:>9.1f} {tpot * 1e3:>9.2f} {tok_per_sec:>10,.0f} {mbu:>7.1f} "
               f"{mfu:>7.2f} {result['peak_vram'] / 2**30:>9.2f} {len(step_times):>6}")
        payload["sweep"].append({
            "batch_size": batch_size, "ttft_sec": round(result["ttft"], 6), "tpot_sec": round(tpot, 6),
            "tok_per_sec": round(tok_per_sec, 1), "mbu_percent": round(mbu, 2), "mfu_percent": round(mfu, 4),
            "peak_vram_bytes": result["peak_vram"], "decode_steps": len(step_times)})
    print0(json.dumps(payload))
    return {"op": "bench", "suite": "infer", **payload}


def _corpus_texts(corpus):
    """First doc batch of the local train and val shards of `corpus`, if downloaded -- the tokenizer
    was trained on train-shard text, so val is the honest one. Missing shards are skipped."""
    from datacore import ParquetDirectorySource
    from tinylab import data
    texts = {}
    train_paths, val_paths = data.corpus_train_val_paths(1, corpus)
    for name, paths in ((f"{corpus}-train", train_paths), (f"{corpus}-val", val_paths)):
        if all(os.path.exists(p) for p in paths):
            _path, batch = next(iter(ParquetDirectorySource(paths=paths).text_batches()))
            texts[name] = "\n".join(batch)
    return texts


def _run_tokenizer(cfg, ctx):
    """suite="tokenizer": compression ratio (bytes per token) of this step's tokenizer on fixed
    texts (news, Korean, code, math, science) plus the local shards of "corpus" (default climbmix) when present, with a
    lossless-roundtrip check. cfg["baselines"] (default none) names tiktoken encodings to compare
    against, e.g. ["gpt2", "cl100k_base"] -- their vocab files are downloaded on first use, so
    leaving it off keeps the suite offline. Returns {"op": "bench", "suite": "tokenizer",
    "vocab_sizes", "results": {tokenizer: {text: {bytes, tokens, ratio}}}}."""
    tokenizers = {"ours": ctx.tokenizer_for(cfg.get("tokenizer"))}
    baselines = cfg.get("baselines", [])
    if baselines:
        import tiktoken
    texts = {**bench_texts.TEXTS, **_corpus_texts(cfg.get("corpus", data_mod.DEFAULT_CORPUS))}
    vocab_sizes, results = {}, {}
    for name in ["ours", *baselines]:
        if name == "ours":
            tok = tokenizers["ours"]
            vocab_sizes[name], encode, decode = tok.get_vocab_size(), tok.encode, tok.decode
        else:
            enc = tiktoken.get_encoding(name)
            vocab_sizes[name], encode, decode = enc.n_vocab, enc.encode_ordinary, enc.decode
        results[name] = {}
        for text_name, text in texts.items():
            ids = encode(text)
            assert decode(ids) == text, f"bench tokenizer: {name!r} does not roundtrip the {text_name!r} text"
            n_bytes = len(text.encode("utf-8"))
            results[name][text_name] = {"bytes": n_bytes, "tokens": len(ids), "ratio": n_bytes / len(ids)}
    print0(f"  {'text':<16} {'bytes':>7} " + " ".join(f"{n + ' tok':>14} {'ratio':>6}" for n in results))
    for text_name in texts:
        cells = " ".join(f"{results[n][text_name]['tokens']:>14,} {results[n][text_name]['ratio']:>6.2f}" for n in results)
        print0(f"  {text_name:<16} {results['ours'][text_name]['bytes']:>7,} {cells}")
    return {"op": "bench", "suite": "tokenizer", "vocab_sizes": vocab_sizes, "results": results}


def run(cfg: dict, ctx) -> dict:
    """Runs one bench step: cfg is a resolved job-file step (see docs/job-file.md for every key),
    ctx the shared Context for this job run."""
    assert "suite" in cfg, f"bench: 'suite' is required (one of {', '.join(SUITES)})"
    suite = cfg["suite"]
    assert suite in SUITES, f"bench: suite must be one of {', '.join(SUITES)}, got {suite!r}"
    if suite == "tokenizer":
        return _run_tokenizer(cfg, ctx)
    if suite == "infer":
        assert ctx.device.type == "cuda", "bench infer: needs a CUDA GPU (timing and VRAM measurement)"
        assert ctx.world_size == 1, "bench infer: a single-GPU benchmark, run without torchrun"
    model, tokenizer, meta = _load(cfg, ctx)
    model.eval()
    if suite == "core":
        return _run_core(cfg, ctx, model, tokenizer)
    if suite == "chat":
        return _run_chat(cfg, ctx, model, tokenizer)
    if suite == "bpb":
        return _run_bpb(cfg, ctx, model, tokenizer)
    if suite == "sample":
        return _run_sample(cfg, ctx, model, tokenizer)
    return _run_infer(cfg, ctx, model, tokenizer, meta)
