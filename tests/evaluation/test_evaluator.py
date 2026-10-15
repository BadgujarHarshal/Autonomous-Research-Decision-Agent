# tests/evaluation/test_evaluator.py

from __future__ import annotations

from types import SimpleNamespace

from app.evaluation.evaluator import (
    AgentEvaluator,
    EvaluationTask,
)


def make_state(
    *,
    status="COMPLETED",
    tool_calls=None,
    tool_results=None,
    final_answer=None,
    errors=None,
):
    return SimpleNamespace(
        status=SimpleNamespace(
            value=status
        ),
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        metadata={
            "final_answer": final_answer
        },
        errors=errors or [],
    )


def test_evaluator_calculation_task():
    task = EvaluationTask(
        task_id="calc_test",
        goal="Calculate 25 * 4",
        expected_tools=[
            "calculate"
        ],
        expected_arguments=[
            {
                "expression": "25 * 4"
            }
        ],
        expected_answer="100",
        category="calculation",
    )

    state = make_state(
        tool_calls=[
            SimpleNamespace(
                tool_name="calculate",
                arguments={
                    "expression": "25 * 4"
                },
            )
        ],
        tool_results=[
            SimpleNamespace(
                tool_name="calculate",
                success=True,
                output=100,
            )
        ],
        final_answer="100",
    )

    evaluator = AgentEvaluator()

    result = evaluator.evaluate_task(
        task,
        state,
    )

    assert (
        result.status
        == "COMPLETED"
    )

    assert (
        result.tool_selection_accuracy
        == 1.0
    )

    assert (
        result.tool_argument_accuracy
        == 1.0
    )

    assert (
        result.final_answer_accuracy
        == 1.0
    )


def test_evaluator_detects_wrong_tool():
    task = EvaluationTask(
        task_id="wrong_tool",
        goal="Calculate 25 * 4",
        expected_tools=[
            "calculate"
        ],
        expected_arguments=[
            {
                "expression": "25 * 4"
            }
        ],
        expected_answer="100",
    )

    state = make_state(
        tool_calls=[
            SimpleNamespace(
                tool_name="search_knowledge",
                arguments={
                    "query": (
                        "Calculate 25 * 4"
                    )
                },
            )
        ],
        tool_results=[
            SimpleNamespace(
                tool_name="search_knowledge",
                success=True,
                output="wrong",
            )
        ],
        final_answer="wrong",
    )

    result = AgentEvaluator().evaluate_task(
        task,
        state,
    )

    assert (
        result.tool_selection_accuracy
        == 0.0
    )

    assert (
        result.final_answer_accuracy
        == 0.0
    )


def test_evaluator_counts_unnecessary_calls():
    task = EvaluationTask(
        task_id="extra_calls",
        goal="Calculate 25 * 4",
        expected_tools=[
            "calculate"
        ],
        expected_arguments=[
            {
                "expression": "25 * 4"
            }
        ],
        expected_answer="100",
    )

    state = make_state(
        tool_calls=[
            SimpleNamespace(
                tool_name="calculate",
                arguments={
                    "expression": "25 * 4"
                },
            ),
            SimpleNamespace(
                tool_name="calculate",
                arguments={
                    "expression": "25 * 4"
                },
            ),
        ],
        tool_results=[
            SimpleNamespace(
                tool_name="calculate",
                success=True,
                output=100,
            ),
            SimpleNamespace(
                tool_name="calculate",
                success=True,
                output=100,
            ),
        ],
        final_answer="100",
    )

    result = AgentEvaluator().evaluate_task(
        task,
        state,
    )

    assert (
        result.unnecessary_tool_call_rate
        == 0.5
    )

    assert (
        result.repeated_tool_call_rate
        == 0.5
    )


def test_evaluator_recovery():
    task = EvaluationTask(
        task_id="recovery",
        goal="Calculate 25 * 4",
        expected_tools=[
            "calculate"
        ],
        expected_arguments=[
            {
                "expression": "25 * 4"
            }
        ],
        expected_answer="100",
        recovery_expected=True,
    )

    state = make_state(
        tool_calls=[
            SimpleNamespace(
                tool_name="calculate",
                arguments={
                    "expression": "25 * 4"
                },
            ),
            SimpleNamespace(
                tool_name="calculate",
                arguments={
                    "expression": "25 * 4"
                },
            ),
        ],
        tool_results=[
            SimpleNamespace(
                tool_name="calculate",
                success=False,
                output=None,
            ),
            SimpleNamespace(
                tool_name="calculate",
                success=True,
                output=100,
            ),
        ],
        final_answer="100",
    )

    result = AgentEvaluator().evaluate_task(
        task,
        state,
    )

    assert (
        result.recovery_success
        == 1.0
    )


def test_evaluator_dataset():
    evaluator = AgentEvaluator()

    task_1 = EvaluationTask(
        task_id="one",
        goal="Calculate 25 * 4",
        expected_tools=[
            "calculate"
        ],
        expected_arguments=[
            {
                "expression": "25 * 4"
            }
        ],
        expected_answer="100",
    )

    state_1 = make_state(
        tool_calls=[
            SimpleNamespace(
                tool_name="calculate",
                arguments={
                    "expression": "25 * 4"
                },
            )
        ],
        tool_results=[
            SimpleNamespace(
                tool_name="calculate",
                success=True,
                output=100,
            )
        ],
        final_answer="100",
    )

    task_2 = EvaluationTask(
        task_id="two",
        goal="Calculate 10 + 5",
        expected_tools=[
            "calculate"
        ],
        expected_arguments=[
            {
                "expression": "10 + 5"
            }
        ],
        expected_answer="15",
    )

    state_2 = make_state(
        tool_calls=[
            SimpleNamespace(
                tool_name="calculate",
                arguments={
                    "expression": "10 + 5"
                },
            )
        ],
        tool_results=[
            SimpleNamespace(
                tool_name="calculate",
                success=True,
                output=15,
            )
        ],
        final_answer="15",
    )

    report = evaluator.evaluate_dataset(
        [
            (task_1, state_1),
            (task_2, state_2),
        ]
    )

    assert (
        report["task_count"]
        == 2
    )

    assert (
        report["metrics"][
            "tool_selection_accuracy"
        ]
        == 1.0
    )

    assert (
        report["metrics"][
            "final_answer_accuracy"
        ]
        == 1.0
    )