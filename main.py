#!/usr/bin/env python3
"""
Gemma 4 Multi-Turn Thinking Chatbot with Streaming & Speculative Decoding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Install: pip install -U transformers torch accelerate

Models used:
  • google/gemma-4-E2B-it          ← target (main) model
  • google/gemma-4-31B-it-assistant← drafter for speculative decoding

Commands during chat:
  quit / exit   → close the chatbot
  reset         → clear conversation history
  think on/off  → toggle thinking mode at runtime
  /help         → show all commands
"""

import sys
import threading
from textwrap import indent

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# ── Model IDs ─────────────────────────────────────────────────────────────────
TARGET_MODEL_ID    = "google/gemma-4-E2B-it"
ASSISTANT_MODEL_ID = "google/gemma-4-31B-it-assistant"

# ── Generation defaults (Gemma 4 recommended) ─────────────────────────────────
MAX_NEW_TOKENS = 2048
TEMPERATURE    = 1.0
TOP_P          = 0.95
TOP_K          = 64

# ── Feature flags ─────────────────────────────────────────────────────────────
ENABLE_THINKING = True   # adds <|think|> to system prompt; enable reasoning
SHOW_THINKING   = True   # print the thought block while streaming

# ── ANSI colours ──────────────────────────────────────────────────────────────
C_THINK   = "\033[2;36m"   # dim cyan   → thinking block
C_ANSWER  = "\033[0;32m"   # green      → final answer
C_LABEL   = "\033[1;33m"   # bold gold  → section labels
C_CMD     = "\033[1;34m"   # bold blue  → prompts/commands
C_ERR     = "\033[1;31m"   # bold red   → errors
C_RESET   = "\033[0m"

# ── Gemma 4 thinking-block delimiters ─────────────────────────────────────────
# The model wraps internal reasoning in:
#   <|channel>thought\n ... <channel|>
THINK_OPEN_TAG  = "<|channel>thought"
THINK_CLOSE_TAG = "<channel|>"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cprint(text: str, color: str = C_RESET, end: str = "\n", flush: bool = False) -> None:
    sys.stdout.write(f"{color}{text}{C_RESET}")
    if end:
        sys.stdout.write(end)
    if flush:
        sys.stdout.flush()


def system_content(enable_thinking: bool) -> str:
    """Build system message. Prepend <|think|> to activate Gemma 4 thinking."""
    base = "You are a helpful, thoughtful assistant."
    return f"<|think|>\n{base}" if enable_thinking else base


def build_messages(history: list[dict], enable_thinking: bool) -> list[dict]:
    """Prepend system turn; history contains only user/assistant messages."""
    return [{"role": "system", "content": system_content(enable_thinking)}] + history


def extract_final_answer(raw: str) -> str:
    """
    Strip the thinking block from the raw model output so only the final
    answer is stored in conversation history (Gemma 4 multi-turn best practice).
    """
    close_idx = raw.find(THINK_CLOSE_TAG)
    if close_idx != -1:
        return raw[close_idx + len(THINK_CLOSE_TAG):].strip()
    return raw.strip()


def show_help() -> None:
    cmds = [
        ("quit / exit",   "End the session"),
        ("reset",         "Clear conversation history"),
        ("think on",      "Enable thinking mode"),
        ("think off",     "Disable thinking mode"),
        ("/help",         "Show this message"),
    ]
    cprint("\nAvailable commands:", C_LABEL)
    for cmd, desc in cmds:
        cprint(f"  {cmd:<18} {desc}", C_CMD)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Streaming renderer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stream_response(streamer: TextIteratorStreamer, show_thinking: bool) -> str:
    """
    Consume the streamer token-by-token, rendering:
      • thinking block in dim cyan  (if show_thinking=True)
      • final answer in green
    Returns the complete raw output string (for history parsing).
    """
    raw_output   = ""
    in_thinking  = False
    buffer       = ""          # lookahead buffer for multi-char delimiters
    answer_begun = False

    # We need to detect THINK_OPEN_TAG and THINK_CLOSE_TAG as they arrive
    # incrementally, so we buffer and flush when a delimiter can't match.
    OPEN  = THINK_OPEN_TAG
    CLOSE = THINK_CLOSE_TAG
    MAX_LOOKAHEAD = max(len(OPEN), len(CLOSE)) - 1

    def flush_buffer(buf: str, in_think: bool) -> None:
        """Print buffered text with appropriate colour."""
        if not buf:
            return
        if in_think and show_thinking:
            cprint(buf, C_THINK, end="", flush=True)
        elif not in_think:
            cprint(buf, C_ANSWER, end="", flush=True)

    for token in streamer:
        raw_output += token
        buffer     += token

        while True:
            if not in_thinking:
                idx = buffer.find(OPEN)
                if idx == 0:
                    # Entering thinking block
                    in_thinking = True
                    if show_thinking:
                        cprint("\n── thinking ──", C_LABEL)
                    buffer = buffer[len(OPEN):]
                elif idx > 0:
                    # Safe text before the opening tag
                    flush_buffer(buffer[:idx], in_think=False)
                    buffer = buffer[idx:]
                    answer_begun = True
                else:
                    # No open tag found yet; check if suffix might start one
                    safe_len = max(0, len(buffer) - MAX_LOOKAHEAD)
                    if safe_len > 0:
                        flush_buffer(buffer[:safe_len], in_think=False)
                        if buffer[:safe_len].strip():
                            answer_begun = True
                        buffer = buffer[safe_len:]
                    break
            else:
                idx = buffer.find(CLOSE)
                if idx == 0:
                    # Leaving thinking block
                    in_thinking = False
                    if show_thinking:
                        cprint("\n── answer ────", C_LABEL)
                    buffer = buffer[len(CLOSE):]
                elif idx > 0:
                    flush_buffer(buffer[:idx], in_think=True)
                    buffer = buffer[idx:]
                else:
                    safe_len = max(0, len(buffer) - MAX_LOOKAHEAD)
                    if safe_len > 0:
                        flush_buffer(buffer[:safe_len], in_think=True)
                        buffer = buffer[safe_len:]
                    break

    # Flush any remaining buffer
    flush_buffer(buffer, in_think=in_thinking)
    print()   # newline after response ends
    return raw_output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_models():
    cprint("⏳  Loading processor …", C_LABEL)
    processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)

    cprint("⏳  Loading target model …", C_LABEL)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    target_model.eval()

    cprint("⏳  Loading drafter (assistant) model …", C_LABEL)
    assistant_model = AutoModelForCausalLM.from_pretrained(
        ASSISTANT_MODEL_ID,
        dtype="auto",
        device_map="auto",
    )
    assistant_model.eval()

    cprint("✅  Models loaded.\n", C_LABEL)
    return processor, target_model, assistant_model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main chat loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chat(processor, target_model, assistant_model) -> None:
    enable_thinking: bool   = ENABLE_THINKING
    show_thinking:   bool   = SHOW_THINKING
    history:         list   = []   # only clean user/assistant turns (no thoughts)

    banner = (
        "╔══════════════════════════════════════════╗\n"
        "║   Gemma 4  ·  Thinking Chatbot  ·  v1.0 ║\n"
        "╚══════════════════════════════════════════╝\n"
        "  Type /help for commands.\n"
    )
    cprint(banner, C_LABEL)

    while True:
        # ── Read user input ───────────────────────────────────────────────────
        try:
            cprint("You: ", C_CMD, end="", flush=True)
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            cprint("\nGoodbye!", C_LABEL)
            break

        if not user_input:
            continue

        # ── Built-in commands ─────────────────────────────────────────────────
        lower = user_input.lower()

        if lower in ("quit", "exit"):
            cprint("Goodbye!", C_LABEL)
            break

        if lower == "reset":
            history = []
            cprint("[History cleared]\n", C_LABEL)
            continue

        if lower == "think on":
            enable_thinking = True
            cprint("[Thinking mode ON]\n", C_LABEL)
            continue

        if lower == "think off":
            enable_thinking = False
            cprint("[Thinking mode OFF]\n", C_LABEL)
            continue

        if lower == "/help":
            show_help()
            continue

        # ── Add user turn & build prompt ──────────────────────────────────────
        history.append({"role": "user", "content": user_input})

        messages = build_messages(history, enable_thinking)

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs   = processor(text=text, return_tensors="pt").to(target_model.device)

        # ── Set up streaming ──────────────────────────────────────────────────
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,   # keep channel/think tokens for parsing
        )

        gen_kwargs = dict(
            **inputs,
            assistant_model=assistant_model,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            streamer=streamer,
        )

        # ── Launch generation thread ──────────────────────────────────────────
        gen_thread = threading.Thread(
            target=target_model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        gen_thread.start()

        # ── Stream & render ───────────────────────────────────────────────────
        cprint("\nAssistant: ", C_LABEL)
        try:
            raw_output = stream_response(streamer, show_thinking=show_thinking)
        except Exception as exc:
            cprint(f"\n[Streaming error: {exc}]", C_ERR)
            raw_output = ""

        gen_thread.join()

        # ── Parse & store only the final answer (no thoughts) ─────────────────
        # Use the processor's built-in parser when thinking is enabled,
        # then fall back to our own extractor for the history.
        if enable_thinking and raw_output:
            try:
                parsed = processor.parse_response(raw_output)
                # parse_response returns a dict with at least a "text" key
                if isinstance(parsed, dict):
                    final_answer = parsed.get("text", "").strip()
                else:
                    final_answer = str(parsed).strip()
            except Exception:
                final_answer = extract_final_answer(raw_output)
        else:
            final_answer = raw_output.strip()

        # Best practice: store ONLY the clean answer in history, never thoughts
        if final_answer:
            history.append({"role": "assistant", "content": final_answer})
        else:
            # Nothing usable came back; pop the user turn to avoid a broken pair
            history.pop()
            cprint("[No response generated — user turn removed from history]\n", C_ERR)
            continue

        print()  # blank line between turns


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    processor, target_model, assistant_model = load_models()
    chat(processor, target_model, assistant_model)
