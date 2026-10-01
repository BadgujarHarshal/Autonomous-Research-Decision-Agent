# app/evaluation/metrics.py

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence


def _normalize_text(value: Any) -> str:
    """
    Normalize a value for deterministic comparison.
    """

    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def exact_match(
    predicted: Any,
    expected: Any,
) -> float:
    """
    Exact normalized string match.

    Returns:
        1.0 if equal, otherwise 0.0.
    """

    return float(
        _normalize_text(predicted)
        == _normalize_text(expected)
    )


def tool_selection_accuracy(
    predicted_tools: Sequence[str],
    expected_tools: Sequence[str],
) -> float:
    """
    Compare the ordered tool trajectory.

    A score of 1.0 means the complete ordered tool sequence
    matches the expected sequence.
    """

    if not expected_tools:
        return 1.0 if not predicted_tools else 0.0

    if not predicted_tools:
        return 0.0

    return float(
        list(predicted_tools)
        == list(expected_tools)
    )


def tool_argument_accuracy(
    predicted_arguments: Sequence[dict[str, Any]],
    expected_arguments: Sequence[dict[str, Any]],
) -> float:
    """
    Calculate the proportion of tool calls whose arguments
    exactly match the expected arguments.
    """

    if not expected_arguments:
        return 1.0 if not predicted_arguments else 0.0

    total = len(expected_arguments)

    matches = 0

    for index, expected in enumerate(expected_arguments):
        if index >= len(predicted_arguments):
            continue

        predicted = predicted_arguments[index]

        if predicted == expected:
            matches += 1

    return matches / total


def execution_success_rate(
    results: Iterable[Any],
) -> float:
    """
    Percentage of executed tools that succeeded.
    """

    results = list(results)

    if not results:
        return 0.0

    successful = sum(
        1
        for result in results
        if bool(getattr(result, "success", False))
    )

    return successful / len(results)


def task_completion_rate(
    statuses: Sequence[str],
) -> float:
    """
    Percentage of tasks ending in COMPLETED.
    """

    if not statuses:
        return 0.0

    completed = sum(
        1
        for status in statuses
        if str(status).upper() == "COMPLETED"
    )

    return completed / len(statuses)


def recovery_success_rate(
    recovered_tasks: int,
    failed_tasks_with_recovery_opportunity: int,
) -> float:
    """
    Recovery success rate.

    Example:
        8 successful recoveries / 10 recovery opportunities = 0.8
    """

    if failed_tasks_with_recovery_opportunity <= 0:
        return 0.0

    return (
        recovered_tasks
        / failed_tasks_with_recovery_opportunity
    )


def unnecessary_tool_call_rate(
    actual_calls: int,
    expected_calls: int,
) -> float:
    """
    Measure excess tool calls relative to the expected trajectory.

    Returns:
        0.0 when there are no unnecessary calls.
    """

    if actual_calls <= 0:
        return 0.0

    if expected_calls < 0:
        raise ValueError(
            "expected_calls cannot be negative."
        )

    unnecessary = max(
        0,
        actual_calls - expected_calls,
    )

    return unnecessary / actual_calls


def repeated_tool_call_rate(
    tool_names: Sequence[str],
) -> float:
    """
    Measure repeated tool invocations.

    A call is considered repeated when the same tool name
    appears more than once in the trajectory.

    Example:
        ["calculate"] -> 0.0

        ["calculate", "calculate"] -> 0.5
    """

    if not tool_names:
        return 0.0

    counts = Counter(tool_names)

    repeated_calls = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated_calls / len(tool_names)


def trajectory_accuracy(
    predicted_tools: Sequence[str],
    expected_tools: Sequence[str],
) -> float:
    """
    Position-aware trajectory accuracy.

    Example:

        expected = [calculate, summarize]
        predicted = [calculate, summarize]

        score = 1.0
    """

    if not expected_tools:
        return 1.0 if not predicted_tools else 0.0

    max_length = max(
        len(predicted_tools),
        len(expected_tools),
    )

    if max_length == 0:
        return 1.0

    matches = 0

    for index in range(
        min(
            len(predicted_tools),
            len(expected_tools),
        )
    ):
        if (
            predicted_tools[index]
            == expected_tools[index]
        ):
            matches += 1

    return matches / max_length


def aggregate_mean(
    values: Iterable[float],
) -> float:
    """
    Calculate the arithmetic mean of numeric values.
    """

    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)


def build_metric_summary(
    *,
    tool_selection: float,
    tool_arguments: float,
    execution_success: float,
    task_completion: float,
    recovery_success: float,
    unnecessary_calls: float,
    repeated_calls: float,
    trajectory: float,
    final_answer: float,
) -> dict[str, float]:
    """
    Return a standardized Phase 5 metric summary.
    """

    return {
        "tool_selection_accuracy": round(
            tool_selection,
            6,
        ),
        "tool_argument_accuracy": round(
            tool_arguments,
            6,
        ),
        "execution_success_rate": round(
            execution_success,
            6,
        ),
        "task_completion_rate": round(
            task_completion,
            6,
        ),
        "recovery_success_rate": round(
            recovery_success,
            6,
        ),
        "unnecessary_tool_call_rate": round(
            unnecessary_calls,
            6,
        ),
        "repeated_tool_call_rate": round(
            repeated_calls,
            6,
        ),
        "trajectory_accuracy": round(
            trajectory,
            6,
        ),
        "final_answer_accuracy": round(
            final_answer,
            6,
        ),
    }