# tests/evaluation/test_metrics.py

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.evaluation.metrics import (
    aggregate_mean,
    build_metric_summary,
    exact_match,
    execution_success_rate,
    recovery_success_rate,
    repeated_tool_call_rate,
    task_completion_rate,
    tool_argument_accuracy,
    tool_selection_accuracy,
    trajectory_accuracy,
    unnecessary_tool_call_rate,
)


def test_exact_match_is_case_and_whitespace_insensitive():
    assert (
        exact_match(
            "  Hello World ",
            "hello world",
        )
        == 1.0
    )


def test_exact_match_returns_zero_for_different_values():
    assert (
        exact_match(
            "hello",
            "world",
        )
        == 0.0
    )


def test_tool_selection_accuracy():
    assert (
        tool_selection_accuracy(
            ["calculate"],
            ["calculate"],
        )
        == 1.0
    )


def test_tool_selection_accuracy_detects_wrong_sequence():
    assert (
        tool_selection_accuracy(
            ["search_knowledge"],
            ["calculate"],
        )
        == 0.0
    )


def test_tool_argument_accuracy():
    assert (
        tool_argument_accuracy(
            [
                {
                    "expression": "25 * 4"
                }
            ],
            [
                {
                    "expression": "25 * 4"
                }
            ],
        )
        == 1.0
    )


def test_tool_argument_accuracy_partial():
    score = tool_argument_accuracy(
        [
            {
                "expression": "25 * 4"
            },
            {
                "expression": "10 + 5"
            },
        ],
        [
            {
                "expression": "25 * 4"
            },
            {
                "expression": "10 - 5"
            },
        ],
    )

    assert score == 0.5


def test_execution_success_rate():
    results = [
        SimpleNamespace(success=True),
        SimpleNamespace(success=True),
        SimpleNamespace(success=False),
    ]

    assert (
        execution_success_rate(
            results
        )
        == pytest.approx(2 / 3)
    )


def test_execution_success_rate_empty():
    assert (
        execution_success_rate([])
        == 0.0
    )


def test_task_completion_rate():
    assert (
        task_completion_rate(
            [
                "COMPLETED",
                "FAILED",
                "COMPLETED",
            ]
        )
        == pytest.approx(2 / 3)
    )


def test_recovery_success_rate():
    assert (
        recovery_success_rate(
            4,
            5,
        )
        == 0.8
    )


def test_recovery_success_rate_without_opportunities():
    assert (
        recovery_success_rate(
            0,
            0,
        )
        == 0.0
    )


def test_unnecessary_tool_call_rate():
    assert (
        unnecessary_tool_call_rate(
            actual_calls=3,
            expected_calls=2,
        )
        == pytest.approx(1 / 3)
    )


def test_no_unnecessary_calls():
    assert (
        unnecessary_tool_call_rate(
            actual_calls=2,
            expected_calls=2,
        )
        == 0.0
    )


def test_repeated_tool_call_rate():
    assert (
        repeated_tool_call_rate(
            [
                "calculate",
                "calculate",
            ]
        )
        == 0.5
    )


def test_repeated_tool_call_rate_without_repetition():
    assert (
        repeated_tool_call_rate(
            [
                "calculate",
                "search_knowledge",
            ]
        )
        == 0.0
    )


def test_trajectory_accuracy():
    assert (
        trajectory_accuracy(
            [
                "calculate",
                "summarize",
            ],
            [
                "calculate",
                "summarize",
            ],
        )
        == 1.0
    )


def test_trajectory_accuracy_partial():
    assert (
        trajectory_accuracy(
            [
                "calculate",
            ],
            [
                "calculate",
                "summarize",
            ],
        )
        == 0.5
    )


def test_aggregate_mean():
    assert (
        aggregate_mean(
            [0.2, 0.4, 0.6]
        )
        == pytest.approx(0.4)
    )


def test_build_metric_summary():
    summary = build_metric_summary(
        tool_selection=1.0,
        tool_arguments=0.9,
        execution_success=0.8,
        task_completion=0.7,
        recovery_success=0.6,
        unnecessary_calls=0.1,
        repeated_calls=0.05,
        trajectory=0.75,
        final_answer=0.85,
    )

    assert (
        summary[
            "tool_selection_accuracy"
        ]
        == 1.0
    )

    assert (
        summary[
            "final_answer_accuracy"
        ]
        == 0.85
    )