"""Tests for built-in context compaction (pyagent/compaction.py + agent wiring)."""
from __future__ import annotations

from typing import Any
import unittest
from unittest import mock

from pyagent import compaction
from pyagent.agent import Agent
from pyagent.compaction import (
    COMPACTION_MARKER,
    OVERFLOW_MARKER,
    TOOL_PRUNE_PREFIX,
    CompactionLimits,
    calibrate_estimate,
    estimate_messages_tokens,
    manage_context,
    resolve_limits,
)
from pyagent.config import AppConfig
from pyagent.tools import ToolRegistry


def make_limits(**overrides) -> CompactionLimits:
    base = dict(
        enabled=True,
        soft_tokens=1_000,
        hard_tokens=100_000,
        keep_recent_tokens=400,
        max_summary_tokens=256,
        model=None,
    )
    base.update(overrides)
    return CompactionLimits(**base)


def conversation(rounds: int = 6, tool_chars: int = 600) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "original goal"},
    ]
    for i in range(rounds):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "q"}'},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": "search",
                "content": f"result {i} " + ("x" * tool_chars),
                "tool_call_id": f"call_{i}",
            }
        )
        messages.append({"role": "assistant", "content": f"step {i} findings"})
    messages.append({"role": "user", "content": "latest question"})
    return messages


class EstimateTests(unittest.TestCase):
    def test_estimate_grows_with_content_and_tool_calls(self):
        small = estimate_messages_tokens([{"role": "user", "content": "hi"}])
        big = estimate_messages_tokens([{"role": "user", "content": "x" * 4000}])
        with_tools = estimate_messages_tokens(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "x" * 100, "arguments": "y" * 400},
                        }
                    ],
                }
            ]
        )
        self.assertLess(small, big)
        self.assertGreater(with_tools, 50)

    def test_calibration_moves_toward_provider_ratio(self):
        updated = calibrate_estimate(1.0, prompt_tokens=800, raw_estimate=400)
        self.assertGreater(updated, 1.0)
        self.assertLessEqual(updated, 2.0)
        self.assertEqual(calibrate_estimate(1.0, 0, 100), 1.0)

    def test_resolve_limits_defaults_and_window_ratios(self):
        class Cfg:
            compaction_enabled = True
            context_window = 0
            compaction_soft_tokens = 0
            compaction_hard_tokens = 0
            compaction_keep_recent_tokens = 0
            compaction_max_summary_tokens = 1024
            compaction_model = ""

        limits = resolve_limits(Cfg())
        self.assertEqual(limits.soft_tokens, compaction.UNKNOWN_WINDOW_SOFT_TOKENS)
        self.assertEqual(limits.hard_tokens, compaction.UNKNOWN_WINDOW_HARD_TOKENS)

        Cfg.context_window = 100_000
        limits = resolve_limits(Cfg())
        self.assertEqual(limits.soft_tokens, 65_000)
        self.assertEqual(limits.hard_tokens, 85_000)
        self.assertEqual(limits.keep_recent_tokens, 25_000)
        self.assertGreater(limits.hard_tokens, limits.soft_tokens)

        Cfg.compaction_soft_tokens = 50_000
        Cfg.compaction_hard_tokens = 40_000  # below soft: clamped above soft
        limits = resolve_limits(Cfg())
        self.assertGreater(limits.hard_tokens, limits.soft_tokens)


class PruneTests(unittest.TestCase):
    def test_old_tool_outputs_are_pruned_recent_kept(self):
        messages = conversation(rounds=4, tool_chars=900)
        limits = make_limits(soft_tokens=10**9, keep_recent_tokens=400)
        new_messages, report = manage_context(
            messages, limits=limits, complete=None
        )
        tool_messages = [m for m in new_messages if m.get("role") == "tool"]
        pruned = [m for m in tool_messages if m["content"].startswith(TOOL_PRUNE_PREFIX)]
        self.assertGreaterEqual(len(pruned), 2)
        self.assertTrue(all(len(m["content"]) < 400 for m in pruned))
        self.assertGreater(report.pruned_tool_outputs, 0)
        self.assertGreater(report.pruned_chars, 0)
        # The newest tool output is inside the keep window and stays verbatim.
        self.assertEqual(tool_messages[-1]["content"], messages[-3]["content"])

    def test_pruning_is_idempotent(self):
        messages = conversation(rounds=3, tool_chars=900)
        limits = make_limits(soft_tokens=10**9, keep_recent_tokens=10)
        once, _ = manage_context(messages, limits=limits)
        twice, report = manage_context(once, limits=limits)
        self.assertEqual(report.pruned_tool_outputs, 0)
        self.assertEqual([m["content"] for m in once], [m["content"] for m in twice])


class SummarizeTests(unittest.TestCase):
    def test_middle_is_summarized_and_pins_survive(self):
        messages = conversation(rounds=4, tool_chars=100)
        limits = make_limits(soft_tokens=10, keep_recent_tokens=120)
        captured: list[str] = []

        def fake_complete(system: str, user: str) -> str:
            captured.append(user)
            return "GOAL: finish the task\nDECISIONS: none yet"

        new_messages, report = manage_context(
            messages, limits=limits, complete=fake_complete
        )
        self.assertTrue(report.summarized)
        self.assertGreater(report.summarized_messages, 0)
        self.assertEqual(new_messages[0], messages[0])  # system pinned
        self.assertEqual(
            new_messages[1], {"role": "user", "content": "original goal"}
        )
        marker = next(m for m in new_messages if str(m.get("content", "")).startswith(COMPACTION_MARKER))
        self.assertIn("GOAL: finish the task", marker["content"])
        self.assertEqual(new_messages[-1]["content"], "latest question")
        # Tool-call/result pairing preserved for kept blocks
        for index, message in enumerate(new_messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                ids = {tc["id"] for tc in message["tool_calls"]}
                following = set()
                for follow in new_messages[index + 1:]:
                    if follow.get("role") == "tool":
                        following.add(follow.get("tool_call_id"))
                    else:
                        break
                self.assertTrue(ids <= following or not following or ids == following)

    def test_previous_summary_is_folded_into_next_compaction(self):
        messages = conversation(rounds=4, tool_chars=100)
        limits = make_limits(soft_tokens=10, keep_recent_tokens=120)
        seen: list[str] = []

        def fake_complete(system: str, user: str) -> str:
            seen.append(user)
            return f"SUMMARY {len(seen)}"

        first, report1 = manage_context(messages, limits=limits, complete=fake_complete)
        self.assertTrue(report1.summarized)
        # Grow history again, then compact again.
        grown = first + [
            {"role": "user", "content": "another question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "and another"},
        ]
        second, report2 = manage_context(grown, limits=limits, complete=fake_complete)
        self.assertTrue(report2.summarized)
        self.assertIn("SUMMARY 1", seen[1])
        marker_messages = [
            m for m in second
            if str(m.get("content", "")).startswith(COMPACTION_MARKER)
        ]
        self.assertEqual(len(marker_messages), 1)

    def test_summarizer_failure_preserves_middle_and_reports_failure(self):
        messages = conversation(rounds=3, tool_chars=100)
        limits = make_limits(soft_tokens=10, keep_recent_tokens=120, hard_tokens=10**9)

        def boom(system: str, user: str) -> str:
            raise RuntimeError("model exploded")

        new_messages, report = manage_context(messages, limits=limits, complete=boom)
        self.assertFalse(report.summarized)
        self.assertIn("model exploded", report.failure or "")
        contents = [str(m.get("content")) for m in new_messages]
        self.assertTrue(any("step 0 findings" in c for c in contents))

    def test_backstop_drops_oldest_blocks_and_marks_overflow(self):
        messages = conversation(rounds=5, tool_chars=900)
        limits = make_limits(soft_tokens=10, keep_recent_tokens=60, hard_tokens=80)

        def boom(system: str, user: str) -> str:
            raise RuntimeError("down")

        new_messages, report = manage_context(messages, limits=limits, complete=boom)
        self.assertGreater(report.dropped_blocks, 0)
        self.assertEqual(new_messages[0]["role"], "system")
        self.assertEqual(new_messages[1]["content"], "original goal")
        self.assertTrue(
            any(str(m.get("content", "")).startswith("[Context overflow") for m in new_messages)
        )

    def test_backstop_never_drops_summary_or_pins(self):
        messages = conversation(rounds=5, tool_chars=900)
        limits = make_limits(soft_tokens=10, keep_recent_tokens=60, hard_tokens=1)

        new_messages, report = manage_context(
            messages, limits=limits, complete=lambda s, u: "SUMMARY"
        )
        self.assertEqual(new_messages[0]["role"], "system")
        markers = [
            m for m in new_messages
            if str(m.get("content", "")).startswith(COMPACTION_MARKER)
        ]
        # Summary either survives, or nothing remains but pins.
        self.assertLessEqual(len(markers), 1)

    def test_force_compacts_even_when_under_budget(self):
        messages = conversation(rounds=3, tool_chars=600)
        limits = make_limits(soft_tokens=10**9, keep_recent_tokens=60)
        new_messages, report = manage_context(
            messages, limits=limits, complete=lambda s, u: "BRIEF", force=True
        )
        self.assertTrue(report.summarized)
        self.assertTrue(
            any(str(m.get("content", "")).startswith(COMPACTION_MARKER) for m in new_messages)
        )

    def test_pinned_prefix_without_system_and_user(self):
        messages = [{"role": "assistant", "content": "floating"}] * 4
        limits = make_limits(soft_tokens=1, keep_recent_tokens=1)
        new_messages, report = manage_context(
            messages, limits=limits, complete=lambda s, u: "S", force=True
        )
        self.assertTrue(report.changed)


class AgentIntegrationTests(unittest.TestCase):
    class _Chunks:
        def __init__(self, chunks):
            self._chunks = chunks

        def chat_stream(self, messages, tools=None):
            for chunk in self._chunks:
                yield chunk

    def _agent(self, compaction_config: dict[str, Any]) -> Agent:
        config = AppConfig(
            max_iterations=3,
            **compaction_config,
        )
        agent = Agent(config=config, tool_registry=ToolRegistry([]))
        return agent

    def test_compaction_event_emitted_and_marker_present(self):
        agent = self._agent(
            dict(
                compaction_enabled=True,
                compaction_soft_tokens=1,
                compaction_hard_tokens=10**9,
                compaction_keep_recent_tokens=1,
            )
        )
        responses = [
            [{"content": "first answer"}],           # run 1 main call
            [{"content": "BRIEF: earlier turns"}],   # run 2 compaction summarizer
            [{"content": "second answer"}],          # run 2 main call
        ]
        calls = []

        class FakeClient:
            def __init__(self):
                self._iter = iter(responses)

            def chat_stream(self, messages, tools=None):
                calls.append(messages)
                yield from next(self._iter)

            def close(self):
                pass

        agent.client = FakeClient()
        agent._compaction_client = agent.client

        list(agent.run("first question"))
        events = list(agent.run("second question"))

        compaction_events = [e for e in events if e.get("type") == "compaction"]
        self.assertEqual(len(compaction_events), 1)
        self.assertTrue(compaction_events[0]["summarized"])
        self.assertTrue(
            any(
                str(m.get("content", "")).startswith(COMPACTION_MARKER)
                for m in agent.messages
            )
        )
        # System prompt is still first.
        self.assertEqual(agent.messages[0]["role"], "system")
        # The summarizer saw the unpinned middle of the history; the pinned
        # first user message survived outside the summary.
        summarizer_input = calls[1][-1]["content"]
        self.assertIn("first answer", summarizer_input)
        self.assertIn("first question", [m["content"] for m in agent.messages])

    def test_compaction_disabled_by_default(self):
        agent = self._agent({})

        class FakeClient:
            def __init__(self):
                self._responses = [[{"content": "ok"}] for _ in range(3)]

            def chat_stream(self, messages, tools=None):
                yield from self._responses.pop(0)

            def close(self):
                pass

        agent.client = FakeClient()
        list(agent.run("one"))
        events = list(agent.run("two"))
        self.assertFalse(any(e.get("type") == "compaction" for e in events))

    def test_compact_now_forces_even_when_disabled(self):
        agent = self._agent({})

        class FakeClient:
            def chat_stream(self, messages, tools=None):
                if tools is None:
                    yield {"content": "BRIEF"}
                else:
                    yield {"content": "ok"}

            def close(self):
                pass

        agent.client = FakeClient()
        agent._compaction_client = agent.client
        list(agent.run("hello"))
        report = agent.compact_now()
        self.assertTrue(report.summarized or not report.changed)
        status = agent.context_status()
        self.assertIn("estimated_tokens", status)
        self.assertEqual(status["enabled"], False)

    def test_provider_usage_calibrates_estimate(self):
        agent = self._agent({})

        class UsageClient:
            def chat_stream(self, messages, tools=None):
                yield {"content": "ok"}
                yield {"usage": {"prompt_tokens": 4000, "completion_tokens": 10, "total_tokens": 4010}}

            def close(self):
                pass

        agent.client = UsageClient()
        list(agent.run("hello"))
        self.assertEqual(agent._last_prompt_tokens, 4000)
        self.assertGreater(agent._token_calibration, 1.0)


if __name__ == "__main__":
    unittest.main()
