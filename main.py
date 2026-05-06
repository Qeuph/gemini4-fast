#!/usr/bin/env python3
"""
Gemma 4 MTP chat CLI.

This script runs a target Gemma 4 model with a Multi-Token Prediction (MTP)
drafter in the Hugging Face Transformers speculative-decoding path. The drafter
predicts several tokens ahead and the target model verifies those candidates in
parallel, preserving the same output distribution as target-only generation while
reducing end-to-end latency on supported hardware.
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass
from queue import Empty
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer

# ── Model IDs ─────────────────────────────────────────────────────────────────
DEFAULT_TARGET_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_MTP_DRAFT_MODEL_ID = "google/gemma-4-31B-it-assistant"

# ── Generation defaults (Gemma-family chat defaults) ──────────────────────────
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_NUM_ASSISTANT_TOKENS = 5
DEFAULT_STREAM_TIMEOUT: float | None = None

# ── Gemma thinking-block delimiters ───────────────────────────────────────────
THINK_OPEN_TAG = "<|channel>thought"
THINK_CLOSE_TAG = "<channel|>"

# ── ANSI colours ──────────────────────────────────────────────────────────────
C_THINK = "\033[2;36m"  # dim cyan   → thinking block
C_ANSWER = "\033[0;32m"  # green      → final answer
C_LABEL = "\033[1;33m"  # bold gold  → section labels
C_CMD = "\033[1;34m"  # bold blue  → prompts/commands
C_ERR = "\033[1;31m"  # bold red   → errors
C_RESET = "\033[0m"


@dataclass(slots=True)
class ChatConfig:
    """Runtime configuration for the chat CLI."""

    target_model_id: str = DEFAULT_TARGET_MODEL_ID
    mtp_draft_model_id: str = DEFAULT_MTP_DRAFT_MODEL_ID
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    num_assistant_tokens: int = DEFAULT_NUM_ASSISTANT_TOKENS
    enable_thinking: bool = True
    show_thinking: bool = True
    enable_mtp: bool = True
    dtype: str = "auto"
    device_map: str = "auto"
    stream_timeout: float | None = DEFAULT_STREAM_TIMEOUT


@dataclass(slots=True)
class LoadedModels:
    """Container for model artifacts used by the chat loop."""

    processor: Any
    target_model: Any
    mtp_draft_model: Any | None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Terminal helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def supports_color() -> bool:
    return sys.stdout.isatty()


def cprint(text: str, color: str = C_RESET, end: str = "\n", flush: bool = False) -> None:
    prefix = color if supports_color() else ""
    suffix = C_RESET if supports_color() else ""
    sys.stdout.write(f"{prefix}{text}{suffix}")
    if end:
        sys.stdout.write(end)
    if flush:
        sys.stdout.flush()


def show_help(config: ChatConfig) -> None:
    cmds = [
        ("quit / exit", "End the session"),
        ("reset", "Clear conversation history"),
        ("think on/off", "Enable or disable thinking mode"),
        ("think show/hide", "Show or hide streamed thinking blocks"),
        ("mtp on/off", "Enable or disable MTP speculative decoding"),
        ("mtp status", "Show target, drafter, and current MTP state"),
        ("tokens <n>", "Change max_new_tokens for future turns"),
        ("/help", "Show this message"),
    ]
    cprint("\nAvailable commands:", C_LABEL)
    for cmd, desc in cmds:
        cprint(f"  {cmd:<18} {desc}", C_CMD)
    print_mtp_status(config)


def print_mtp_status(config: ChatConfig) -> None:
    state = "ON" if config.enable_mtp else "OFF"
    cprint("\nMTP speculative decoding:", C_LABEL)
    cprint(f"  state:               {state}", C_CMD)
    cprint(f"  target model:        {config.target_model_id}", C_CMD)
    cprint(f"  MTP draft model:     {config.mtp_draft_model_id}", C_CMD)
    cprint(f"  draft tokens/step:   {config.num_assistant_tokens}\n", C_CMD)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt and response helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def system_content(enable_thinking: bool) -> str:
    """Build a system message and opt in to model thinking when requested."""
    base = "You are a helpful, thoughtful assistant."
    return f"<|think|>\n{base}" if enable_thinking else base


def build_messages(history: list[dict[str, str]], enable_thinking: bool) -> list[dict[str, str]]:
    """Prepend a system turn; history contains only user/assistant messages."""
    return [{"role": "system", "content": system_content(enable_thinking)}] + history


def extract_final_answer(raw: str) -> str:
    """Strip Gemma thinking markup before storing assistant messages in history."""
    close_idx = raw.find(THINK_CLOSE_TAG)
    if close_idx != -1:
        return raw[close_idx + len(THINK_CLOSE_TAG) :].strip()
    return raw.strip()


def parse_final_answer(processor: Any, raw_output: str, enable_thinking: bool) -> str:
    """Use processor parsing when available, with a delimiter-based fallback."""
    if not raw_output:
        return ""
    if not enable_thinking:
        return raw_output.strip()

    parse_response = getattr(processor, "parse_response", None)
    if callable(parse_response):
        try:
            parsed = parse_response(raw_output)
            if isinstance(parsed, dict):
                return str(parsed.get("text", "")).strip()
            return str(parsed).strip()
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    return extract_final_answer(raw_output)


def render_chunk(buf: str, in_think: bool, show_thinking: bool) -> None:
    """Print buffered response text with the correct colour and visibility."""
    if not buf:
        return
    if in_think and show_thinking:
        cprint(buf, C_THINK, end="", flush=True)
    elif not in_think:
        cprint(buf, C_ANSWER, end="", flush=True)


def stream_response(streamer: TextIteratorStreamer, show_thinking: bool) -> str:
    """
    Consume streamed text, render thinking separately from final answers, and
    return the raw model output for history parsing.
    """
    raw_output = ""
    in_thinking = False
    buffer = ""

    max_lookahead = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG)) - 1

    try:
        for token in streamer:
            raw_output += token
            buffer += token

            while True:
                if not in_thinking:
                    idx = buffer.find(THINK_OPEN_TAG)
                    if idx == 0:
                        in_thinking = True
                        if show_thinking:
                            cprint("\n── thinking ──", C_LABEL)
                        buffer = buffer[len(THINK_OPEN_TAG) :]
                    elif idx > 0:
                        render_chunk(buffer[:idx], in_think=False, show_thinking=show_thinking)
                        buffer = buffer[idx:]
                    else:
                        safe_len = max(0, len(buffer) - max_lookahead)
                        if safe_len > 0:
                            render_chunk(buffer[:safe_len], in_think=False, show_thinking=show_thinking)
                            buffer = buffer[safe_len:]
                        break
                else:
                    idx = buffer.find(THINK_CLOSE_TAG)
                    if idx == 0:
                        in_thinking = False
                        if show_thinking:
                            cprint("\n── answer ────", C_LABEL)
                        buffer = buffer[len(THINK_CLOSE_TAG) :]
                    elif idx > 0:
                        render_chunk(buffer[:idx], in_think=True, show_thinking=show_thinking)
                        buffer = buffer[idx:]
                    else:
                        safe_len = max(0, len(buffer) - max_lookahead)
                        if safe_len > 0:
                            render_chunk(buffer[:safe_len], in_think=True, show_thinking=show_thinking)
                            buffer = buffer[safe_len:]
                        break
    except Empty as exc:
        raise RuntimeError("timed out waiting for streamed model output") from exc

    render_chunk(buffer, in_think=in_thinking, show_thinking=show_thinking)
    print()
    return raw_output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model loading and generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_models(config: ChatConfig) -> LoadedModels:
    """Load the target model and, when enabled, its MTP drafter."""
    cprint("⏳  Loading processor …", C_LABEL)
    processor = AutoProcessor.from_pretrained(config.target_model_id)

    cprint("⏳  Loading target model …", C_LABEL)
    target_model = AutoModelForCausalLM.from_pretrained(
        config.target_model_id,
        dtype=config.dtype,
        device_map=config.device_map,
    )
    target_model.eval()

    mtp_draft_model = None
    if config.enable_mtp:
        cprint("⏳  Loading MTP drafter model …", C_LABEL)
        mtp_draft_model = AutoModelForCausalLM.from_pretrained(
            config.mtp_draft_model_id,
            dtype=config.dtype,
            device_map=config.device_map,
        )
        mtp_draft_model.eval()

    cprint("✅  Models loaded.", C_LABEL)
    print_mtp_status(config)
    return LoadedModels(processor, target_model, mtp_draft_model)


def model_input_device(target_model: Any) -> torch.device | str:
    """Pick the input device without assuming a single-GPU model layout."""
    device = getattr(target_model, "device", None)
    if device is not None:
        return device
    for parameter in target_model.parameters():
        return parameter.device
    return "cpu"


def build_generation_kwargs(
    config: ChatConfig,
    loaded: LoadedModels,
    inputs: Any,
    streamer: TextIteratorStreamer,
) -> dict[str, Any]:
    """Create generation kwargs and attach the MTP drafter when active."""
    gen_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "do_sample": config.temperature > 0,
        "streamer": streamer,
    }

    if config.enable_mtp and loaded.mtp_draft_model is not None:
        gen_kwargs["assistant_model"] = loaded.mtp_draft_model
        gen_kwargs["num_assistant_tokens"] = config.num_assistant_tokens

    return gen_kwargs


def run_generation_worker(
    target_model: Any,
    gen_kwargs: dict[str, Any],
    streamer: TextIteratorStreamer,
    errors: list[Exception],
) -> None:
    """Run generation in a worker thread and unblock the streamer on failure."""
    try:
        with torch.inference_mode():
            target_model.generate(**gen_kwargs)
    except Exception as exc:
        errors.append(exc)
        streamer.end()


def generate_turn(
    config: ChatConfig,
    loaded: LoadedModels,
    history: list[dict[str, str]],
) -> str:
    """Generate and stream a single assistant turn."""
    messages = build_messages(history, config.enable_thinking)
    text = loaded.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config.enable_thinking,
    )

    inputs = loaded.processor(text=text, return_tensors="pt").to(model_input_device(loaded.target_model))
    tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        timeout=config.stream_timeout,
    )
    gen_kwargs = build_generation_kwargs(config, loaded, inputs, streamer)
    generation_errors: list[Exception] = []

    gen_thread = threading.Thread(
        target=run_generation_worker,
        args=(loaded.target_model, gen_kwargs, streamer, generation_errors),
        daemon=True,
    )
    gen_thread.start()

    cprint("\nAssistant: ", C_LABEL)
    raw_output = stream_response(streamer, show_thinking=config.show_thinking)
    gen_thread.join()
    if generation_errors:
        raise RuntimeError(str(generation_errors[0])) from generation_errors[0]
    return raw_output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command handling and chat loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def handle_command(
    user_input: str,
    config: ChatConfig,
    history: list[dict[str, str]],
    loaded: LoadedModels,
) -> bool:
    """Handle slash-style and plain-text commands. Returns True if consumed."""
    lower = user_input.lower()

    if lower == "reset":
        history.clear()
        cprint("[History cleared]\n", C_LABEL)
        return True

    if lower == "think on":
        config.enable_thinking = True
        cprint("[Thinking mode ON]\n", C_LABEL)
        return True

    if lower == "think off":
        config.enable_thinking = False
        cprint("[Thinking mode OFF]\n", C_LABEL)
        return True

    if lower == "think show":
        config.show_thinking = True
        cprint("[Thinking blocks will be shown]\n", C_LABEL)
        return True

    if lower == "think hide":
        config.show_thinking = False
        cprint("[Thinking blocks will be hidden]\n", C_LABEL)
        return True

    if lower == "mtp on":
        if loaded.mtp_draft_model is None:
            cprint(
                "[MTP drafter was not loaded. Restart without --disable-mtp to enable it.]\n",
                C_ERR,
            )
        else:
            config.enable_mtp = True
            cprint("[MTP speculative decoding ON]\n", C_LABEL)
        return True

    if lower == "mtp off":
        config.enable_mtp = False
        cprint("[MTP speculative decoding OFF]\n", C_LABEL)
        return True

    if lower == "mtp status":
        print_mtp_status(config)
        return True

    if lower.startswith("tokens "):
        _, value = user_input.split(maxsplit=1)
        if value.isdigit() and int(value) > 0:
            config.max_new_tokens = int(value)
            cprint(f"[max_new_tokens set to {config.max_new_tokens}]\n", C_LABEL)
        else:
            cprint("[Usage: tokens <positive integer>]\n", C_ERR)
        return True

    if lower == "/help":
        show_help(config)
        return True

    return False


def chat(config: ChatConfig, loaded: LoadedModels) -> None:
    history: list[dict[str, str]] = []

    banner = (
        "╔══════════════════════════════════════════════════════╗\n"
        "║ Gemma 4 · MTP Speculative Decoding Chatbot · v2.0  ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
        "  Type /help for commands. MTP is enabled by default.\n"
    )
    cprint(banner, C_LABEL)

    while True:
        try:
            cprint("You: ", C_CMD, end="", flush=True)
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            cprint("\nGoodbye!", C_LABEL)
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            cprint("Goodbye!", C_LABEL)
            break

        if handle_command(user_input, config, history, loaded):
            continue

        history.append({"role": "user", "content": user_input})

        try:
            raw_output = generate_turn(config, loaded, history)
        except RuntimeError as exc:
            history.pop()
            cprint(f"\n[Generation error: {exc}]", C_ERR)
            cprint("[User turn removed from history]\n", C_ERR)
            continue

        final_answer = parse_final_answer(loaded.processor, raw_output, config.enable_thinking)
        if final_answer:
            history.append({"role": "assistant", "content": final_answer})
        else:
            history.pop()
            cprint("[No response generated — user turn removed from history]\n", C_ERR)
            continue

        print()


def parse_args(argv: list[str] | None = None) -> ChatConfig:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 chat with an MTP drafter via speculative decoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL_ID)
    parser.add_argument("--mtp-draft-model", "--assistant-model", default=DEFAULT_MTP_DRAFT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-assistant-tokens", type=int, default=DEFAULT_NUM_ASSISTANT_TOKENS)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=DEFAULT_STREAM_TIMEOUT,
        help="Seconds to wait between streamed chunks before treating generation as stalled; disabled by default.",
    )
    parser.add_argument("--no-thinking", action="store_true", help="Disable Gemma thinking prompts.")
    parser.add_argument("--hide-thinking", action="store_true", help="Do not print thinking blocks while streaming.")
    parser.add_argument("--disable-mtp", action="store_true", help="Run target-only generation without the MTP drafter.")

    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.num_assistant_tokens <= 0:
        parser.error("--num-assistant-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.stream_timeout is not None and args.stream_timeout <= 0:
        parser.error("--stream-timeout must be positive when set")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the interval (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")

    return ChatConfig(
        target_model_id=args.target_model,
        mtp_draft_model_id=args.mtp_draft_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_assistant_tokens=args.num_assistant_tokens,
        enable_thinking=not args.no_thinking,
        show_thinking=not args.hide_thinking,
        enable_mtp=not args.disable_mtp,
        dtype=args.dtype,
        device_map=args.device_map,
        stream_timeout=args.stream_timeout,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    loaded = load_models(config)
    chat(config, loaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
