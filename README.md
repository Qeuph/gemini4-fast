# gemini4-fast

`gemini4-fast` is a small command-line chatbot for Gemma 4 instruction models that uses a Multi-Token Prediction (MTP) drafter through Transformers speculative decoding.

The target model remains responsible for the final distribution and verifies the draft tokens in parallel, so MTP keeps standard-generation quality while reducing latency on compatible hardware. The repo defaults to:

- **Target model:** `google/gemma-4-E2B-it`
- **MTP drafter:** `google/gemma-4-31B-it-assistant`

> This project is intended for environments where you have accepted the model licenses, have access to the checkpoints, and have enough CPU/GPU memory for both the target and draft models.

## What MTP does here

Multi-Token Prediction (MTP) extends the base generation pipeline with a smaller, faster draft model. In each decoding step, the drafter predicts several future tokens. The target model then verifies those tokens in parallel. Accepted draft tokens are emitted immediately; rejected tokens fall back to target-model generation.

In this repo, MTP is enabled by default by passing the loaded drafter as `assistant_model` to `target_model.generate(...)`. You can disable it with `--disable-mtp` for a target-only baseline or during a chat with `mtp off`. If you start with `--disable-mtp`, the drafter is not loaded; restart without that flag before using `mtp on`.

## Features

- Streaming terminal chat UI.
- MTP speculative decoding enabled by default.
- Runtime MTP controls: `mtp status`, `mtp on`, and `mtp off`.
- Runtime thinking controls: `think on`, `think off`, `think show`, and `think hide`.
- Conversation reset without restarting the process.
- CLI flags for model IDs, sampling settings, dtype, device mapping, draft-token count, and optional stream timeouts.
- Clean conversation history: thinking blocks are stripped before assistant turns are stored.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using gated Hugging Face model repositories, log in before running:

```bash
huggingface-cli login
```

## Run

```bash
python main.py
```

The app loads the processor, target model, and MTP drafter, then starts an interactive chat.

### Common options

```bash
python main.py \
  --target-model google/gemma-4-E2B-it \
  --mtp-draft-model google/gemma-4-31B-it-assistant \
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

Use a specific device map or dtype:

```bash
python main.py --device-map auto --dtype auto
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
| `think on` / `think off` | Enable or disable thinking prompts for future turns. |
| `think show` / `think hide` | Show or hide streamed thinking blocks. |
| `mtp status` | Print the target model, draft model, and draft-token setting. |
| `mtp on` / `mtp off` | Enable or disable use of the already-loaded MTP drafter. |
| `tokens <n>` | Change `max_new_tokens` for future turns. |
| `/help` | Show the command list. |

## How MTP is wired

The relevant generation path is:

1. Load `AutoProcessor` from the target model.
2. Load the target `AutoModelForCausalLM`.
3. Load the MTP draft `AutoModelForCausalLM`.
4. Build generation kwargs.
5. Attach the drafter with `assistant_model=<mtp_draft_model>` and set `num_assistant_tokens`.
6. Call `target_model.generate(...)` with a `TextIteratorStreamer`.

Because the target model verifies drafted tokens, the resulting text should match the quality of standard target-model decoding while improving throughput when the drafter is significantly faster.

## Benchmarking MTP

To compare latency on your machine, run the same prompt twice:

```bash
python main.py --disable-mtp
python main.py --num-assistant-tokens 5
```

Use realistic prompts and long enough generations to measure decode throughput. The best `--num-assistant-tokens` value depends on model sizes, hardware, quantization, batch size, and prompt shape. Start with `5`, then try values such as `3`, `8`, or `12`.

## Troubleshooting

- **Out of memory:** use a smaller target or drafter, lower `--max-new-tokens`, or adjust `--device-map`/quantization outside this minimal script.
- **Model access errors:** accept the model terms on Hugging Face and run `huggingface-cli login`.
- **No speedup:** speculative decoding helps most when the drafter is much faster than the target and draft-token acceptance is high. Try tuning `--num-assistant-tokens`.
- **Unexpected thinking text in history:** the app strips Gemma thinking delimiters before saving assistant turns, but raw streamed output keeps special tokens visible so the renderer can separate thinking from answers.
- **Streaming appears stuck:** generation failures in the worker thread are surfaced back to the chat loop. For long-running remote or overloaded environments, set `--stream-timeout <seconds>` to fail a turn if no streamed chunks arrive before the timeout.

## Development checks

```bash
python -m compileall main.py
python main.py --help
```
