"""
`python -m tinylab chat <job.json>` -- interactive chat, driven by a job file's sibling "chat"
block rather than its own flags (the same file that trained the model already names its base dir,
checkpoint source and tag). This is deliberately a separate command, not a job op: an op runs to
completion unattended as part of a pipeline, and a REPL blocks on a human at a prompt -- see
AGENTS.md.

Conversation rendering follows nanochat's scripts/chat_cli.py convention (llmllab/nanochat),
driving the same tinylab.engine.Engine the `bench` op hands to benchcore as its Generator, so the
interactive path and the scored path exercise identical decoding code.
"""
from tinylab import checkpoints, job
from tinylab.engine import Engine
from tinylab.ops import COMMON_KEYS, Context
from tinylab.runtime import compute_cleanup, print0

ACCEPTED_KEYS = {"source", "model_tag", "model_step", "temperature", "top_k", "max_tokens", "prompt"}


def main(job_path: str):
    j = job.load(job_path)
    cfg = job.resolve_chat(j)
    job.check_known_keys(cfg, ACCEPTED_KEYS | COMMON_KEYS, where="\"chat\"")
    assert "model_tag" in cfg, "chat: \"chat\": {\"model_tag\": ...} is required in the job file"

    ctx = Context(device_type=cfg.get("device", "auto"))
    source = cfg.get("source", "sft")
    model, tokenizer, meta = checkpoints.load_model(source, ctx.device, phase="eval", model_tag=cfg["model_tag"], step=cfg.get("model_step"))
    model.eval()
    engine = Engine(model, tokenizer, manager=ctx.model_manager)

    temperature = cfg.get("temperature", 0.6)
    top_k = cfg.get("top_k", 50)
    max_tokens = cfg.get("max_tokens", 256)

    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

    def _one_turn(conversation_tokens, user_input):
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)
        # Print incrementally as bytes, not token-by-token as text: a single token's bytes aren't
        # always valid UTF-8 on their own (a multi-byte character can split across a token
        # boundary), so decoding each token in isolation can emit replacement characters for
        # perfectly valid text. `pending` holds bytes not yet resolved into a complete character.
        pending = b""
        for token_column, _masks in engine.generate(conversation_tokens, num_samples=1, max_tokens=max_tokens,
                                                      temperature=temperature, top_k=top_k):
            token = token_column[0]
            conversation_tokens.append(token)
            if token == assistant_end:
                break
            pending += tokenizer.decode_single_token_bytes(token)
            try:
                text = pending.decode("utf-8")
            except UnicodeDecodeError:
                continue  # wait for the rest of this character's bytes
            print(text, end="", flush=True)
            pending = b""
        if pending:
            print(pending.decode("utf-8", errors="replace"), end="", flush=True)
        print()
        if conversation_tokens[-1] != assistant_end:
            conversation_tokens.append(assistant_end)
        return conversation_tokens

    print0(f"Loaded {source}:{cfg['model_tag']} (step {meta.get('step')}). Ctrl-D or an empty line to exit.")
    try:
        conversation_tokens = [bos]
        if "prompt" in cfg:
            _one_turn(conversation_tokens, cfg["prompt"])
            return
        while True:
            try:
                user_input = input("\nuser> ")
            except EOFError:
                print()
                break
            if not user_input.strip():
                break
            print("assistant> ", end="", flush=True)
            conversation_tokens = _one_turn(conversation_tokens, user_input)
    finally:
        compute_cleanup()
