# app/evaluation/evaluator.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from app.evaluation.metrics import (
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


@dataclass
class EvaluationTask:
    """
    Expected behavior for one evaluation task.
    """

    task_id: str
    goal: str
    expected_tools: list[str] = field(
        default_factory=list
    )
    expected_arguments: list[dict[str, Any]] = field(
        default_factory=list
    )
    expected_answer: Optional[str] = None
    category: str = "general"
    difficulty: str = "easy"
    recovery_expected: bool = False


@dataclass
class TaskEvaluationResult:
    """
    Evaluation result for a single task.
    """

    task_id: str
    goal: str

    status: str

    predicted_tools: list[str]
    predicted_arguments: list[dict[str, Any]]

    expected_tools: list[str]
    expected_arguments: list[dict[str, Any]]

    predicted_answer: Optional[str]
    expected_answer: Optional[str]

    tool_selection_accuracy: float
    tool_argument_accuracy: float
    execution_success_rate: float
    task_completion: float
    recovery_success: float
    unnecessary_tool_call_rate: float
    repeated_tool_call_rate: float
    trajectory_accuracy: float
    final_answer_accuracy: float

    error_count: int
    tool_call_count: int


class AgentEvaluator:
    """
    Evaluates an AgentState produced by the Phase 3/4 agent.

    The evaluator does not modify the state and does not influence
    execution.
    """

    def evaluate_task(
        self,
        task: EvaluationTask,
        state: Any,
    ) -> TaskEvaluationResult:
        """
        Evaluate one completed agent state.
        """

        predicted_tools = [
            str(
                getattr(
                    call,
                    "tool_name",
                    "",
                )
            )
            for call in getattr(
                state,
                "tool_calls",
                [],
            )
        ]

        predicted_arguments = [
            dict(
                getattr(
                    call,
                    "arguments",
                    {},
                )
                or {}
            )
            for call in getattr(
                state,
                "tool_calls",
                [],
            )
        ]

        results = list(
            getattr(
                state,
                "tool_results",
                [],
            )
        )

        status = str(
            getattr(
                getattr(
                    state,
                    "status",
                    "",
                ),
                "value",
                getattr(
                    state,
                    "status",
                    "",
                ),
            )
        ).upper()

        predicted_answer = None

        metadata = getattr(
            state,
            "metadata",
            {},
        )

        if isinstance(metadata, dict):
            value = metadata.get(
                "final_answer"
            )

            if value is not None:
                predicted_answer = str(value)

        expected_answer_accuracy = 0.0

        if task.expected_answer is None:
            expected_answer_accuracy = (
                1.0
                if predicted_answer is not None
                else 0.0
            )
        else:
            expected_answer_accuracy = exact_match(
                predicted_answer,
                task.expected_answer,
            )

        recovered = (
            task.recovery_expected
            and status == "COMPLETED"
            and len(results) > 1
            and any(
                not bool(
                    getattr(
                        result,
                        "success",
                        False,
                    )
                )
                for result in results
            )
            and any(
                bool(
                    getattr(
                        result,
                        "success",
                        False,
                    )
                )
                for result in results
            )
        )

        recovery_opportunity = (
            1
            if task.recovery_expected
            else 0
        )

        recovery_score = recovery_success_rate(
            1 if recovered else 0,
            recovery_opportunity,
        )

        return TaskEvaluationResult(
            task_id=task.task_id,
            goal=task.goal,
            status=status,
            predicted_tools=predicted_tools,
            predicted_arguments=predicted_arguments,
            expected_tools=list(
                task.expected_tools
            ),
            expected_arguments=[
                dict(arguments)
                for arguments
                in task.expected_arguments
            ],
            predicted_answer=predicted_answer,
            expected_answer=task.expected_answer,
            tool_selection_accuracy=tool_selection_accuracy(
                predicted_tools,
                task.expected_tools,
            ),
            tool_argument_accuracy=tool_argument_accuracy(
                predicted_arguments,
                task.expected_arguments,
            ),
            execution_success_rate=execution_success_rate(
                results,
            ),
            task_completion=float(
                status == "COMPLETED"
            ),
            recovery_success=recovery_score,
            unnecessary_tool_call_rate=unnecessary_tool_call_rate(
                len(predicted_tools),
                len(task.expected_tools),
            ),
            repeated_tool_call_rate=repeated_tool_call_rate(
                predicted_tools,
            ),
            trajectory_accuracy=trajectory_accuracy(
                predicted_tools,
                task.expected_tools,
            ),
            final_answer_accuracy=expected_answer_accuracy,
            error_count=len(
                getattr(
                    state,
                    "errors",
                    [],
                )
            ),
            tool_call_count=len(
                predicted_tools
            ),
        )

    def evaluate_dataset(
        self,
        evaluated_tasks: Sequence[
            tuple[EvaluationTask, Any]
        ],
    ) -> dict[str, Any]:
        """
        Evaluate an entire dataset.
        """

        results = [
            self.evaluate_task(
                task,
                state,
            )
            for task, state
            in evaluated_tasks
        ]

        if not results:
            return {
                "task_count": 0,
                "results": [],
                "metrics": {},
            }

        metrics = {
            "tool_selection_accuracy": self._mean(
                result.tool_selection_accuracy
                for result in results
            ),
            "tool_argument_accuracy": self._mean(
                result.tool_argument_accuracy
                for result in results
            ),
            "execution_success_rate": self._mean(
                result.execution_success_rate
                for result in results
            ),
            "task_completion_rate": self._mean(
                result.task_completion
                for result in results
            ),
            "recovery_success_rate": self._mean(
                result.recovery_success
                for result in results
                if result.expected_answer is not None
                or result.recovery_success > 0
            ),
            "unnecessary_tool_call_rate": self._mean(
                result.unnecessary_tool_call_rate
                for result in results
            ),
            "repeated_tool_call_rate": self._mean(
                result.repeated_tool_call_rate
                for result in results
            ),
            "trajectory_accuracy": self._mean(
                result.trajectory_accuracy
                for result in results
            ),
            "final_answer_accuracy": self._mean(
                result.final_answer_accuracy
                for result in results
            ),
        }

        return {
            "task_count": len(results),
            "metrics": {
                key: round(
                    value,
                    6,
                )
                for key, value in metrics.items()
            },
            "results": [
                self._serialize_result(
                    result
                )
                for result in results
            ],
        }

    @staticmethod
    def _mean(
        values,
    ) -> float:
        values = list(values)

        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _serialize_result(
        result: TaskEvaluationResult,
    ) -> dict[str, Any]:
        return {
            "task_id": result.task_id,
            "goal": result.goal,
            "status": result.status,
            "predicted_tools": result.predicted_tools,
            "predicted_arguments": result.predicted_arguments,
            "expected_tools": result.expected_tools,
            "expected_arguments": result.expected_arguments,
            "predicted_answer": result.predicted_answer,
            "expected_answer": result.expected_answer,
            "tool_selection_accuracy": result.tool_selection_accuracy,
            "tool_argument_accuracy": result.tool_argument_accuracy,
            "execution_success_rate": result.execution_success_rate,
            "task_completion": result.task_completion,
            "recovery_success": result.recovery_success,
            "unnecessary_tool_call_rate": (
                result.unnecessary_tool_call_rate
            ),
            "repeated_tool_call_rate": (
                result.repeated_tool_call_rate
            ),
            "trajectory_accuracy": (
                result.trajectory_accuracy
            ),
            "final_answer_accuracy": (
                result.final_answer_accuracy
            ),
            "error_count": result.error_count,
            "tool_call_count": result.tool_call_count,
        }