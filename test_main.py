import io
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout

import main


class FakeProcessor:
    def __init__(self, parsed):
        self.parsed = parsed

    def parse_response(self, raw):
        return self.parsed


class ChatRenderingTests(unittest.TestCase):
    def test_parse_response_falls_back_when_processor_returns_empty_text(self):
        raw = f"{main.THINK_OPEN_TAG}hidden{main.THINK_CLOSE_TAG}Hello!{main.GENERATED_SPECIAL_TOKENS[0]}"
        self.assertEqual(main.parse_final_answer(FakeProcessor({"text": ""}), raw, True), "Hello!")

    def test_extract_final_answer_strips_thinking_and_turn_markers(self):
        raw = f"Before {main.THINK_OPEN_TAG}hidden{main.THINK_CLOSE_TAG}After<turn|>"
        self.assertEqual(main.extract_final_answer(raw), "Before After")

    def test_stream_response_hides_markers_and_tps_by_default(self):
        chunks = iter([
            main.THINK_OPEN_TAG[:8],
            main.THINK_OPEN_TAG[8:] + "hidden",
            main.THINK_CLOSE_TAG + "Hello",
            " there<turn|>",
        ])
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            raw = main.stream_response(
                chunks,
                tokenizer=object(),
                show_thinking=False,
                show_tps=False,
                stop_event=threading.Event(),
            )

        self.assertIn("Hello there", stdout.getvalue())
        self.assertNotIn("<turn|>", stdout.getvalue())
        self.assertNotIn("hidden", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("hidden", raw)


if __name__ == "__main__":
    unittest.main()
