# Gemma 4 MTP Chat CLI

`gemma4-fast` is a small command-line chatbot for Gemma 4 instruction models that uses a Multi-Token Prediction (MTP) drafter through Transformers speculative decoding.

The target model remains responsible for the final distribution and verifies the draft tokens in parallel, so MTP keeps standard-generation quality while reducing latency on compatible hardware. The repo defaults to:

- **Target model:** `google/gemma-4-E2B-it`
- **MTP drafter:** `google/gemma-4-E2B-it-assistant`

> This project is intended for environments where you have accepted the model licenses, have access to the checkpoints, and have enough CPU/GPU memory for both the target and draft models.

## What MTP does here

Multi-Token Prediction (MTP) extends the base generation pipeline with a smaller, faster draft model. In each decoding step, the drafter predicts several future tokens. The target model then verifies those tokens in parallel. Accepted draft tokens are emitted immediately; rejected tokens fall back to target-model generation.

In this repo, MTP is enabled by default by passing the loaded drafter as `assistant_model` to `target_model.generate(...)`. You can disable it with `--disable-mtp` for a target-only baseline or during a chat with `mtp off`. If you start with `--disable-mtp`, the drafter is not loaded; restart without that flag before using `mtp on`.

## Features

- Streaming terminal chat UI with Unix `readline` line editing and command history.
- Multi-line prompts: end a line with `\` and continue typing on the next line.
- MTP speculative decoding enabled by default.
- Runtime MTP controls: `mtp status`, `mtp on`, and `mtp off`.
- Runtime thinking controls: `think on`, `think off`, `think show`, and `think hide`.
- Conversation reset and history truncation without restarting the process.
- Live token-per-second display on interactive terminals.
- Graceful Ctrl+C handling during generation: the current stream is cancelled and the chat prompt returns.
- CLI flags for model IDs, sampling settings, dtype, device mapping, draft-token count, history size, and optional stream timeouts.
- Clean conversation history: thinking blocks are stripped before assistant turns are stored.
- Friendly model-loading errors with common Hugging Face access and device-placement suggestions.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `transformers>=4.42` because speculative decoding and MTP-related generation behavior depends on relatively recent Transformers releases. The base install includes `accelerate`, `huggingface_hub`, `transformers`, and `torch`. Optional quantization packages such as `bitsandbytes` are not installed by default; add them separately if you adapt the script for quantized loading.

If you are using gated Hugging Face model repositories, log in before running:

```bash
huggingface-cli login
```

## Run

```bash
python main.py
```

The app loads the processor, target model, and MTP drafter, then starts an interactive chat. A small spinner is shown during each model-loading stage when stdout is attached to a TTY.

### Common options

```bash
python main.py \
  --target-model google/gemma-4-E2B-it \
  --mtp-draft-model google/gemma-4-E2B-it-assistant \
  --num-assistant-tokens 5 \
  --max-new-tokens 1024
```

Disable MTP for a baseline comparison:

```bash
python main.py --disable-mtp
```

Hide streamed thinking blocks while still keeping thinking enabled:

```bash
python main.py --hide-thinking
```

Limit stored conversation context to the most recent 8 turns:

```bash
python main.py --max-history-turns 8
```

Use a specific device map or dtype:

```bash
python main.py --device-map auto --dtype auto
```

If model loading fails with automatic placement, try sequential placement:

```bash
python main.py --device-map sequential
```

Treat stalled streaming as an error after 120 seconds without a chunk:

```bash
python main.py --stream-timeout 120
```

## In-chat commands

| Command | Description |
| --- | --- |
| `quit` / `exit` | End the session. |
| `reset` | Clear conversation history. |
| `truncate <n>` | Keep only the last `n` user/assistant turns in history. |
| `truncate` | Use the `--max-history-turns` value to trim history, if one was configured. |
| `think on` / `think off` | Enable or disable thinking prompts for future turns. |
| `think show` / `think hide` | Show or hide streamed thinking blocks. |
| `mtp status` | Print the target model, draft model, draft-token setting, and history limit. |
| `mtp on` / `mtp off` | Enable or disable use of the already-loaded MTP drafter. |
| `tokens <n>` | Change `max_new_tokens` for future turns. |
| `/help` | Show the command list. |

### Multi-line input

For multi-paragraph prompts, put a backslash at the end of each line that should continue:

```text
You: Summarize this passage:\
...  Paragraph one.\
...  Paragraph two.
```

The CLI joins the entered lines with newline characters and sends them as one user turn.

## Thinking-block handling

Gemma chat templates can add model-specific thinking tokens when `apply_chat_template(..., enable_thinking=True)` is used. The CLI intentionally leaves those tokens out of the system prompt and lets the processor template add the right control tokens.

Raw streamed output is kept intact while rendering so the terminal can color or hide thinking text separately from the final answer. Before an assistant message is stored in conversation history, the CLI cleans it:

- all complete thinking blocks are removed, including multiple blocks in one response;
- if a thinking block opens but never closes, the unterminated thinking portion is discarded and a warning is printed;
- the cleaned final answer is what future turns see in history.

This prevents hidden reasoning markup from polluting later prompts while still making live rendering understandable.

## How MTP is wired

The relevant generation path is:

1. Load `AutoProcessor` from the target model.
2. Load the target `AutoModelForCausalLM`.
3. Load the MTP draft `AutoModelForCausalLM` when MTP is enabled.
4. Cache constant generation kwargs such as sampling settings.
5. Reuse a resettable `TextIteratorStreamer` across sequential turns.
6. Build per-turn inputs and attach the streamer.
7. Attach the drafter with `assistant_model=<mtp_draft_model>` and set `num_assistant_tokens`.
8. Call `target_model.generate(...)` in a worker thread with a cancellation-aware stopping criterion.

Only the foreground stream-consumer writes assistant text to stdout; the generation worker only fills the streamer. If you add async logging later, keep terminal writes coordinated so status output does not interleave with streamed tokens.

Because the target model verifies drafted tokens, the resulting text should match the quality of standard target-model decoding while improving throughput when the drafter is significantly faster.

## Tuning `--num-assistant-tokens`

`--num-assistant-tokens` controls how many draft tokens the assistant model proposes per speculative step. Larger values can improve speed if the target accepts many draft tokens, but they also use more VRAM and may lower the acceptance rate when the drafter guesses too far ahead. Start with the default `5`, compare against `--disable-mtp`, then try nearby values such as `3`, `8`, or `12` with prompts that resemble your real workload.

## Benchmarking MTP

To compare latency on your machine, run the same prompt twice:

```bash
python main.py --disable-mtp
python main.py --num-assistant-tokens 5
```

Use realistic prompts and long enough generations to measure decode throughput. The live token-per-second indicator gives a quick interactive read, but for rigorous benchmarking you should still run repeated trials and compare end-to-end latency.

## Troubleshooting

- **Out of memory:** use a smaller target or drafter, lower `--max-new-tokens`, cap history with `--max-history-turns`, or adjust `--device-map`/quantization outside this minimal script.
- **Model access errors:** accept the model terms on Hugging Face and run `huggingface-cli login`.
- **Automatic device placement fails:** try `--device-map sequential` or a smaller model pair.
- **No speedup:** speculative decoding helps most when the drafter is much faster than the target and draft-token acceptance is high. Try tuning `--num-assistant-tokens`.
- **Unexpected thinking text in history:** raw streamed output keeps special tokens visible for the renderer, but saved assistant turns are sanitized before being reused.
- **Streaming appears stuck:** generation failures in the worker thread are surfaced back to the chat loop. For long-running remote or overloaded environments, set `--stream-timeout <seconds>` to fail a turn if no streamed chunks arrive before the timeout.
- **Ctrl+C during generation:** the CLI requests generation cancellation, ends the current stream, removes the interrupted user turn from history, and returns to the prompt instead of exiting the process.

## Development checks

```bash
python -m compileall main.py
python main.py --help
```
