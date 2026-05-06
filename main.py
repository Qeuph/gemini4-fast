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
import itertools
try:
    import readline  # Enables Unix line editing and command history for input().
except ImportError:
    readline = None
import sys
import termios
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

# ── Model IDs ─────────────────────────────────────────────────────────────────
DEFAULT_TARGET_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_MTP_DRAFT_MODEL_ID = "google/gemma-4-E2B-it-assistant"

# ── Generation defaults (Gemma-family chat defaults) ──────────────────────────
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_NUM_ASSISTANT_TOKENS = 5
DEFAULT_STREAM_TIMEOUT: float | None = None
DEFAULT_MAX_HISTORY_TURNS: int | None = None
DEFAULT_SHOW_THINKING = True
DEFAULT_SHOW_TPS = True

# ── Gemma thinking-block delimiters ───────────────────────────────────────────
THINK_OPEN_TAG = "<|channel>thought"
THINK_CLOSE_TAG = "<channel|>"
GENERATED_SPECIAL_TOKENS = (
    "<turn|>",
    "<end_of_turn>",
    "<eos>",
)

# ── ANSI colours ──────────────────────────────────────────────────────────────
C_THINK = "\033[1;36m"  # bold cyan  → thinking block
C_ANSWER = "\033[0m"     # reset      → final answer (use default text color)
C_LABEL = "\033[1;33m"  # bold gold  → section labels
C_CMD = "\033[1;34m"  # bold blue  → prompts/commands
C_ERR = "\033[1;31m"  # bold red   → errors
C_RESET = "\033[0m"


class GenerationInterrupted(Exception):
    """Raised when the user cancels an in-flight streaming generation."""


class StopOnEvent(StoppingCriteria):
    """Stop generation as soon as the controlling thread asks for cancellation."""

    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        return self.stop_event.is_set()


class ReusableTextIteratorStreamer(TextIteratorStreamer):
    """TextIteratorStreamer with an explicit reset hook for sequential chat turns."""

    def reset(self) -> None:
        self.text_queue = Queue()
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True


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
    show_thinking: bool = DEFAULT_SHOW_THINKING
    enable_mtp: bool = True
    dtype: str = "auto"
    device_map: str = "auto"
    stream_timeout: float | None = DEFAULT_STREAM_TIMEOUT
    max_history_turns: int | None = DEFAULT_MAX_HISTORY_TURNS
    show_tps: bool = DEFAULT_SHOW_TPS


@dataclass(slots=True)
class LoadedModels:
    """Container for model artifacts used by the chat loop."""

    processor: Any
    target_model: Any
    mtp_draft_model: Any | None
    streamer: ReusableTextIteratorStreamer
    base_generation_kwargs: dict[str, Any] = field(default_factory=dict)


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


def ensure_terminal_echo() -> None:
    """Best-effort restore of stdin echo before prompting for user input."""
    # In some environments like Google Colab, standard termios calls might fail
    # or not be necessary, but we try anyway.
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
        if not (attrs[3] & termios.ECHO):
            attrs[3] |= termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    except Exception:
        # If termios fails, we can't do much more for echo, but we shouldn't crash.
        pass

    # Additionally, some notebook environments might need a manual reset if they've
    # been messed up by interrupted processes.
    if sys.stdout.isatty():
        sys.stdout.write("\033[m")  # Reset all attributes
        sys.stdout.flush()


def show_help(config: ChatConfig) -> None:
    cmds = [
        ("quit / exit", "End the session"),
        ("reset", "Clear conversation history"),
        ("truncate [n]", "Keep only the last n turns (or configured default)"),
        ("think on/off", "Enable or disable thinking mode"),
        ("think show/hide", "Show or hide streamed thinking blocks"),
        ("mtp on/off", "Enable or disable MTP speculative decoding"),
        ("mtp status", "Show target, drafter, and current MTP state"),
        ("tokens <n>", "Change max_new_tokens for future turns"),
        ("tps on/off", "Show or hide live token-per-second stats"),
        ("/help", "Show this message"),
    ]
    cprint("\nAvailable commands:", C_LABEL)
    for cmd, desc in cmds:
        cprint(f"  {cmd:<18} {desc}", C_CMD)
    cprint("\nInput tip: end a line with \\ to continue a multi-line prompt.", C_CMD)
    print_mtp_status(config)


def print_mtp_status(config: ChatConfig) -> None:
    state = "ON" if config.enable_mtp else "OFF"
    max_history = "unlimited" if config.max_history_turns is None else str(config.max_history_turns)
    tps_state = "ON" if config.show_tps else "OFF"
    cprint("\nMTP speculative decoding:", C_LABEL)
    cprint(f"  state:               {state}", C_CMD)
    cprint(f"  target model:        {config.target_model_id}", C_CMD)
    cprint(f"  MTP draft model:     {config.mtp_draft_model_id}", C_CMD)
    cprint(f"  draft tokens/step:   {config.num_assistant_tokens}", C_CMD)
    cprint(f"  max history turns:   {max_history}", C_CMD)
    cprint(f"  live TPS stats:      {tps_state}\n", C_CMD)


def format_loading_error(exc: Exception, config: ChatConfig) -> str:
    """Return a concise loading error with actionable recovery suggestions."""
    suggestions = [
        "accept gated model terms on Hugging Face if required",
        "run `huggingface-cli login` for private or gated checkpoints",
        "verify --target-model and --mtp-draft-model are accessible model IDs",
        "try `--device-map sequential` or a smaller model if automatic placement fails",
        "try a smaller dtype/quantized setup if you hit CPU or GPU memory limits",
    ]
    details = str(exc).strip() or exc.__class__.__name__
    return (
        "Model loading failed.\n"
        f"  target model:    {config.target_model_id}\n"
        f"  MTP draft model: {config.mtp_draft_model_id if config.enable_mtp else '(disabled)'}\n"
        f"  device map:      {config.device_map}\n"
        f"  dtype:           {config.dtype}\n"
        f"  error:           {details}\n\n"
        "Suggestions:\n  - " + "\n  - ".join(suggestions)
    )


class LoadingSpinner:
    """Small terminal spinner for model-loading steps that lack progress callbacks."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._enabled = sys.stdout.isatty()

    def __enter__(self) -> LoadingSpinner:
        if self._enabled:
            self._thread.start()
        else:
            cprint(f"⏳  {self.message} …", C_LABEL)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._done.set()
        if self._enabled:
            self._thread.join()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _spin(self) -> None:
        for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._done.is_set():
                break
            sys.stdout.write(f"\r{frame}  {self.message} …")
            sys.stdout.flush()
            time.sleep(0.1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt and response helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def system_content(enable_thinking: bool) -> str:
    """
    Build a system message.

    Thinking is controlled by apply_chat_template(enable_thinking=...), which
    owns the model-specific special tokens. Do not manually prepend <|think|>.
    """
    return "You are a helpful, thoughtful assistant."


def build_messages(history: list[dict[str, str]], enable_thinking: bool) -> list[dict[str, str]]:
    """Prepend a system turn; history contains only user/assistant messages."""
    return [{"role": "system", "content": system_content(enable_thinking)}] + history


def strip_generated_special_tokens(text: str) -> str:
    """Remove generated turn/end markers that should not be shown or saved."""
    cleaned = text
    for token in GENERATED_SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def extract_final_answer(raw: str) -> str:
    """Strip all complete or dangling Gemma thinking blocks from assistant history."""
    output_parts: list[str] = []
    cursor = 0

    while cursor < len(raw):
        open_idx = raw.find(THINK_OPEN_TAG, cursor)
        if open_idx == -1:
            output_parts.append(raw[cursor:])
            break

        output_parts.append(raw[cursor:open_idx])
        think_start = open_idx + len(THINK_OPEN_TAG)
        close_idx = raw.find(THINK_CLOSE_TAG, think_start)
        if close_idx == -1:
            cprint(
                "[Warning: unterminated thinking block discarded from saved history]",
                C_ERR,
            )
            break
        cursor = close_idx + len(THINK_CLOSE_TAG)

    return strip_generated_special_tokens("".join(output_parts))


def parse_final_answer(processor: Any, raw_output: str, enable_thinking: bool) -> str:
    """Use processor parsing when available, then sanitize thinking markup."""
    if not raw_output:
        return ""
    if not enable_thinking:
        return strip_generated_special_tokens(raw_output)

    parse_response = getattr(processor, "parse_response", None)
    if callable(parse_response):
        try:
            parsed = parse_response(raw_output)
            if isinstance(parsed, dict):
                parsed_text = next(
                    (str(parsed[key]).strip() for key in ("text", "content", "answer") if parsed.get(key)),
                    "",
                )
            else:
                parsed_text = str(parsed).strip()
            if parsed_text and THINK_OPEN_TAG not in parsed_text and THINK_CLOSE_TAG not in parsed_text:
                return strip_generated_special_tokens(parsed_text)
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


def find_first_marker(buffer: str, markers: tuple[str, ...]) -> tuple[int, str] | None:
    """Return the earliest complete marker found in the current stream buffer."""
    matches = ((idx, marker) for marker in markers if (idx := buffer.find(marker)) != -1)
    return min(matches, default=None, key=lambda item: item[0])


def render_tps(token_count: int, started_at: float, show_tps: bool, final: bool = False) -> None:
    """Render a live token-per-second indicator on a dedicated bottom line."""
    if not show_tps:
        return

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    tps = token_count / elapsed

    # ANSI escape codes:
    # \033[s   : Save cursor position
    # \033[u   : Restore cursor position
    # \033[K   : Clear line from cursor to end
    # \033[1B  : Move cursor down 1 line
    # \033[1A  : Move cursor up 1 line

    if final:
        # On final, we just print it normally on a new line and stay there.
        sys.stdout.write(f"\n{C_LABEL}[~{tps:6.2f} tok/s, {token_count} tokens]{C_RESET}\n")
        sys.stdout.flush()
    else:
        # For live updates: save position, move to next line, print, restore position.
        # We use stdout instead of stderr to ensure they share the same buffer/position.
        sys.stdout.write("\033[s")  # Save cursor
        sys.stdout.write("\n\033[K")  # Move down and clear line
        sys.stdout.write(f"{C_LABEL}[~{tps:6.2f} tok/s, {token_count} tokens]{C_RESET}")
        sys.stdout.write("\033[u")  # Restore cursor
        sys.stdout.flush()


def estimate_token_count(tokenizer: Any, text: str) -> int:
    """Approximate token count for finalized streamer text."""
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return max(1, len(encode(text, add_special_tokens=False)))
        except (TypeError, ValueError):
            pass
    return 1 if text else 0


def stream_response(
    streamer: TextIteratorStreamer,
    tokenizer: Any,
    show_thinking: bool,
    show_tps: bool,
    stop_event: threading.Event,
) -> str:
    """
    Consume streamed text, render thinking separately from final answers, and
    return the raw model output for history parsing.
    """
    raw_output = ""
    in_thinking = False
    buffer = ""
    token_count = 0
    started_at = time.perf_counter()
    hidden_markers = (THINK_OPEN_TAG, THINK_CLOSE_TAG, *GENERATED_SPECIAL_TOKENS)
    max_lookahead = max(len(marker) for marker in hidden_markers) - 1

    try:
        for token in streamer:
            token_count += estimate_token_count(tokenizer, token)
            raw_output += token
            buffer += token
            render_tps(token_count, started_at, show_tps)

            while True:
                marker_match = find_first_marker(buffer, hidden_markers)
                if marker_match is None:
                    safe_len = max(0, len(buffer) - max_lookahead)
                    if safe_len > 0:
                        render_chunk(buffer[:safe_len], in_think=in_thinking, show_thinking=show_thinking)
                        buffer = buffer[safe_len:]
                    break

                idx, marker = marker_match
                if idx > 0:
                    render_chunk(buffer[:idx], in_think=in_thinking, show_thinking=show_thinking)
                    buffer = buffer[idx:]
                    continue

                if marker == THINK_OPEN_TAG:
                    in_thinking = True
                    if show_thinking:
                        cprint("\n💭 Thinking...", C_LABEL)
                elif marker == THINK_CLOSE_TAG:
                    in_thinking = False
                    if show_thinking:
                        cprint("\n\n✨ Answer:", C_LABEL)

                buffer = buffer[len(marker) :]
    except KeyboardInterrupt as exc:
        stop_event.set()
        streamer.end()
        raise GenerationInterrupted from exc
    except Empty as exc:
        raise RuntimeError("timed out waiting for streamed model output") from exc
    finally:
        # Render any remaining text in the buffer before finishing.
        render_chunk(strip_generated_special_tokens(buffer), in_think=in_thinking, show_thinking=show_thinking)
        if show_tps:
            render_tps(token_count, started_at, show_tps, final=True)
        else:
            print()

    return raw_output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model loading and generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def make_base_generation_kwargs(config: ChatConfig) -> dict[str, Any]:
    """Cache generation settings that stay constant across turns."""
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "do_sample": config.temperature > 0,
    }


def load_models(config: ChatConfig) -> LoadedModels:
    """Load the target model and, when enabled, its MTP drafter."""
    try:
        with LoadingSpinner("Loading processor"):
            processor = AutoProcessor.from_pretrained(config.target_model_id)

        with LoadingSpinner("Loading target model"):
            target_model = AutoModelForCausalLM.from_pretrained(
                config.target_model_id,
                dtype=config.dtype,
                device_map=config.device_map,
            )
            target_model.eval()

        mtp_draft_model = None
        if config.enable_mtp:
            with LoadingSpinner("Loading MTP drafter model"):
                mtp_draft_model = AutoModelForCausalLM.from_pretrained(
                    config.mtp_draft_model_id,
                    dtype=config.dtype,
                    device_map=config.device_map,
                )
                mtp_draft_model.eval()
    except Exception as exc:
        raise RuntimeError(format_loading_error(exc, config)) from exc

    cprint("✅  Models loaded.", C_LABEL)
    print_mtp_status(config)
    tokenizer = getattr(processor, "tokenizer", processor)
    streamer = ReusableTextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        timeout=config.stream_timeout,
    )
    return LoadedModels(
        processor=processor,
        target_model=target_model,
        mtp_draft_model=mtp_draft_model,
        streamer=streamer,
        base_generation_kwargs=make_base_generation_kwargs(config),
    )


def model_input_device(target_model: Any) -> torch.device | str:
    """Pick the input device without assuming a single-GPU model layout."""
    device = getattr(target_model, "device", None)
    if device is not None:
        return device

    input_embeddings = getattr(target_model, "get_input_embeddings", lambda: None)()
    embedding_weight = getattr(input_embeddings, "weight", None)
    embedding_device = getattr(embedding_weight, "device", None)
    if embedding_device is not None:
        return embedding_device

    for parameter in target_model.parameters():
        return parameter.device
    return "cpu"


def build_generation_kwargs(
    config: ChatConfig,
    loaded: LoadedModels,
    inputs: Any,
    streamer: TextIteratorStreamer,
    stopping_criteria: StoppingCriteriaList,
) -> dict[str, Any]:
    """Create generation kwargs and attach the MTP drafter when active."""
    gen_kwargs: dict[str, Any] = {
        **loaded.base_generation_kwargs,
        **inputs,
        "max_new_tokens": config.max_new_tokens,
        "streamer": streamer,
        "stopping_criteria": stopping_criteria,
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
    streamer = loaded.streamer
    streamer.timeout = config.stream_timeout
    streamer.reset()
    stop_event = threading.Event()
    stopping_criteria = StoppingCriteriaList([StopOnEvent(stop_event)])
    gen_kwargs = build_generation_kwargs(config, loaded, inputs, streamer, stopping_criteria)
    generation_errors: list[Exception] = []

    gen_thread = threading.Thread(
        target=run_generation_worker,
        args=(loaded.target_model, gen_kwargs, streamer, generation_errors),
        daemon=True,
    )
    gen_thread.start()

    cprint("\n" + "─" * 40, C_LABEL)
    cprint("Assistant:", C_LABEL)
    try:
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        raw_output = stream_response(
            streamer,
            tokenizer=tokenizer,
            show_thinking=config.show_thinking,
            show_tps=config.show_tps,
            stop_event=stop_event,
        )
    except (GenerationInterrupted, KeyboardInterrupt):
        stop_event.set()
        gen_thread.join(timeout=2)
        if gen_thread.is_alive():
            # If thread won't die, we have to recreate the streamer to unblock queues
            tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
            loaded.streamer = ReusableTextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,
                timeout=config.stream_timeout,
            )
        cprint("\n\n[Generation interrupted]", C_ERR)
        return ""
    except Exception as exc:
        stop_event.set()
        gen_thread.join(timeout=2)
        cprint(f"\n\n[Error during generation: {exc}]", C_ERR)
        return ""

    gen_thread.join(timeout=5)
    if generation_errors:
        raise RuntimeError(str(generation_errors[0])) from generation_errors[0]
    return raw_output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command handling and chat loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def truncate_history(history: list[dict[str, str]], max_turns: int | None) -> int:
    """Keep only the most recent user turns and associated assistant messages."""
    if max_turns is None or max_turns <= 0:
        return 0

    user_turns_seen = 0
    keep_start = 0
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "user":
            user_turns_seen += 1
            if user_turns_seen == max_turns:
                keep_start = idx
                break
    else:
        keep_start = 0

    if user_turns_seen < max_turns:
        return 0

    removed = keep_start
    if removed > 0:
        del history[:keep_start]
    return removed


def parse_positive_int(value: str) -> int | None:
    return int(value) if value.isdigit() and int(value) > 0 else None


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

    if lower == "truncate" or lower.startswith("truncate "):
        parts = user_input.split(maxsplit=1)
        if len(parts) == 2:
            max_turns = parse_positive_int(parts[1])
            if max_turns is None:
                cprint("[Usage: truncate [positive integer]]\n", C_ERR)
                return True
        elif config.max_history_turns is not None:
            max_turns = config.max_history_turns
        else:
            cprint("[Usage: truncate <positive integer> or start with --max-history-turns]\n", C_ERR)
            return True

        removed = truncate_history(history, max_turns)
        cprint(f"[History truncated to last {max_turns} turn(s); removed {removed} message(s)]\n", C_LABEL)
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
        max_tokens = parse_positive_int(value)
        if max_tokens is not None:
            config.max_new_tokens = max_tokens
            cprint(f"[max_new_tokens set to {config.max_new_tokens}]\n", C_LABEL)
        else:
            cprint("[Usage: tokens <positive integer>]\n", C_ERR)
        return True

    if lower == "tps on":
        config.show_tps = True
        cprint("[Live token-per-second stats ON]\n", C_LABEL)
        return True

    if lower == "tps off":
        config.show_tps = False
        cprint("[Live token-per-second stats OFF]\n", C_LABEL)
        return True

    if lower == "/help":
        show_help(config)
        return True

    return False


def read_multiline_input() -> str:
    """Read one prompt, continuing when a line ends with a backslash."""
    lines: list[str] = []
    prompt = "You: "

    while True:
        ensure_terminal_echo()
        # In Colab/Notebooks, input() usually handles its own echo.
        # The issue might be coming from readline if it's not well-supported.
        cprint(prompt, C_CMD, end="", flush=True)
        try:
            line = input()
        except EOFError:
            raise
        except KeyboardInterrupt:
            # If they hit Ctrl+C at the prompt, just clear the line and start over.
            print()
            lines = []
            prompt = "You: "
            continue

        if line.endswith("\\"):
            lines.append(line[:-1])
            prompt = "...  "
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def chat(config: ChatConfig, loaded: LoadedModels) -> None:
    history: list[dict[str, str]] = []

    banner = (
        "\033[1;36m"
        "╭──────────────────────────────────────────────────────╮\n"
        "│  Gemma 4 · MTP Speculative Decoding Chatbot · v2.1   │\n"
        "╰──────────────────────────────────────────────────────╯\n"
        "\033[0m"
        "  Type \033[1;34m/help\033[0m for commands. MTP, Thinking, and TPS are \033[1;32mON\033[0m.\n"
    )
    cprint(banner, C_LABEL)

    while True:
        try:
            user_input = read_multiline_input()
        except EOFError:
            cprint("\nGoodbye!", C_LABEL)
            break
        except KeyboardInterrupt:
            # This is already handled in read_multiline_input for the prompt,
            # but we catch it here just in case.
            print()
            continue

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            cprint("Goodbye!", C_LABEL)
            break

        if handle_command(user_input, config, history, loaded):
            continue

        history.append({"role": "user", "content": user_input})
        truncate_history(history, config.max_history_turns)

        try:
            raw_output = generate_turn(config, loaded, history)
        except RuntimeError as exc:
            history.pop()
            cprint(f"\n[Generation error: {exc}]", C_ERR)
            cprint("[User turn removed from history]\n", C_ERR)
            continue

        if not raw_output:
            history.pop()
            cprint("[User turn removed from history]\n", C_ERR)
            continue

        final_answer = parse_final_answer(loaded.processor, raw_output, config.enable_thinking)
        if final_answer:
            history.append({"role": "assistant", "content": final_answer})
            truncate_history(history, config.max_history_turns)
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
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=DEFAULT_MAX_HISTORY_TURNS,
        help="Maximum user/assistant turns to keep in context; disabled by default.",
    )
    parser.add_argument("--no-thinking", action="store_true", help="Disable Gemma thinking prompts.")
    parser.add_argument("--hide-thinking", action="store_true", help="Hide thinking blocks while streaming.")
    parser.add_argument("--disable-mtp", action="store_true", help="Run target-only generation without the MTP drafter.")
    parser.add_argument(
        "--hide-tps",
        action="store_true",
        help="Hide live token-per-second stats while streaming.",
    )

    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.num_assistant_tokens <= 0:
        parser.error("--num-assistant-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.stream_timeout is not None and args.stream_timeout <= 0:
        parser.error("--stream-timeout must be positive when set")
    if args.max_history_turns is not None and args.max_history_turns <= 0:
        parser.error("--max-history-turns must be positive when set")
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
        max_history_turns=args.max_history_turns,
        show_tps=not args.hide_tps,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        loaded = load_models(config)
    except RuntimeError as exc:
        cprint(f"\n{exc}\n", C_ERR)
        return 1
    chat(config, loaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
