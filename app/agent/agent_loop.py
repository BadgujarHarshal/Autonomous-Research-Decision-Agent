# app/agent/agent_loop.py

from __future__ import annotations

from typing import Optional

from app.agent.controller import AgentController
from app.agent.schemas import (
    AgentStatus,
    Observation,
    ToolResult,
)
from app.agent.state import AgentState
from app.tools.executor import ToolExecutor


class AgentLoop:
    """
    Main autonomous execution loop.

    Phase 4 adds:
    - bounded retries
    - re-planning after retry exhaustion
    - recovery from tool failures
    - stopping controls
    - repeated execution protection
    - explicit execution trace metadata
    """

    def __init__(
        self,
        controller: AgentController,
        executor: ToolExecutor,
        max_retries_per_step: int = 2,
        max_replans: int = 3,
        stop_on_unrecoverable_error: bool = True,
    ) -> None:
        if max_retries_per_step < 0:
            raise ValueError(
                "max_retries_per_step cannot be negative."
            )

        if max_replans < 0:
            raise ValueError(
                "max_replans cannot be negative."
            )

        self.controller = controller
        self.executor = executor
        self.max_retries_per_step = (
            max_retries_per_step
        )
        self.max_replans = max_replans
        self.stop_on_unrecoverable_error = (
            stop_on_unrecoverable_error
        )

    def run(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Execute the agent until completion, failure,
        or a configured stopping condition.
        """

        if state.plan is None:
            raise ValueError(
                "AgentState must contain a plan before execution."
            )

        state.set_status(
            AgentStatus.EXECUTING
        )

        while True:
            if not self.controller.should_continue(
                state
            ):
                if state.has_remaining_steps():
                    self._stop_for_limit(
                        state
                    )
                else:
                    state.set_status(
                        AgentStatus.COMPLETED
                    )

                break

            decision = self.controller.decide(
                state
            )

            state.add_decision(
                decision
            )

            if decision.action == "finish":
                self._handle_finish_decision(
                    state,
                    decision,
                )
                break

            if decision.action != "execute":
                state.add_error(
                    (
                        "Unsupported controller action: "
                        f"{decision.action}"
                    )
                )

                state.set_status(
                    AgentStatus.FAILED
                )
                break

            if decision.tool_call is None:
                state.add_error(
                    (
                        "Controller returned execute "
                        "action without a ToolCall."
                    )
                )

                state.set_status(
                    AgentStatus.FAILED
                )
                break

            state.add_tool_call(
                decision.tool_call
            )

            result = self.executor.execute(
                decision.tool_call
            )

            state.add_tool_result(
                result
            )

            self._record_observation(
                state=state,
                result=result,
            )

            if result.success:
                self._handle_success(
                    state,
                    result,
                )
                continue

            should_stop = self._handle_failure(
                state,
                result,
            )

            if should_stop:
                break

        if state.status == AgentStatus.COMPLETED:
            final_answer = (
                self.controller._build_final_answer(
                    state
                )
            )

            if final_answer:
                state.metadata[
                    "final_answer"
                ] = final_answer

        state.metadata[
            "replan_count"
        ] = state.replan_count

        state.metadata[
            "retry_count"
        ] = state.retry_count

        state.metadata[
            "total_tool_calls"
        ] = len(state.tool_calls)

        state.metadata[
            "total_errors"
        ] = len(state.errors)

        return state

    def _handle_finish_decision(
        self,
        state: AgentState,
        decision,
    ) -> None:
        if decision.final_answer:
            state.metadata[
                "final_answer"
            ] = decision.final_answer

        if state.errors:
            state.set_status(
                AgentStatus.FAILED
            )
        else:
            state.set_status(
                AgentStatus.COMPLETED
            )

    def _record_observation(
        self,
        state: AgentState,
        result: ToolResult,
    ) -> None:
        current_step = (
            state.current_step()
        )

        if current_step is None:
            step_id = max(
                1,
                state.current_step_index + 1,
            )
        else:
            step_id = current_step.step_id

        observation = Observation(
            step_id=step_id,
            source=result.tool_name,
            content=(
                result.output
                if result.success
                else result.error
            ),
            success=result.success,
            metadata={
                "latency_ms": result.latency_ms,
            },
        )

        state.add_observation(
            observation
        )

    def _handle_success(
        self,
        state: AgentState,
        result: ToolResult,
    ) -> None:
        current_step = (
            state.current_step()
        )

        if current_step is not None:
            current_step.status = (
                "success"
            )
            current_step.result = result

        state.retry_count = 0

        state.advance_step()

        if not state.has_remaining_steps():
            state.set_status(
                AgentStatus.COMPLETED
            )

    def _handle_failure(
        self,
        state: AgentState,
        result: ToolResult,
    ) -> bool:
        current_step = (
            state.current_step()
        )

        if current_step is not None:
            current_step.status = (
                "failed"
            )
            current_step.result = result

        error_message = (
            result.error
            or f"Tool '{result.tool_name}' failed."
        )

        state.add_error(
            error_message
        )

        # ---------------------------------------------------------
        # Recovery level 1: retry
        # ---------------------------------------------------------
        if (
            state.retry_count
            < self.max_retries_per_step
        ):
            state.increment_retry()

            state.set_status(
                AgentStatus.EXECUTING
            )

            return False

        # ---------------------------------------------------------
        # Recovery level 2: replan
        # ---------------------------------------------------------
        if self.controller.can_replan(
            state,
            self.max_replans,
        ):
            self.controller.mark_replan(
                state
            )

            state.retry_count = 0

            state.set_status(
                AgentStatus.REPLANNING
            )

            self._replan_after_failure(
                state
            )

            state.set_status(
                AgentStatus.EXECUTING
            )

            return False

        # ---------------------------------------------------------
        # Recovery level 3: terminal failure
        # ---------------------------------------------------------
        if self.stop_on_unrecoverable_error:
            state.set_status(
                AgentStatus.FAILED
            )

            return True

        state.set_status(
            AgentStatus.ABORTED
        )

        return True

    def _replan_after_failure(
        self,
        state: AgentState,
    ) -> None:
        """
        Baseline replanning strategy.

        Phase 4 does not yet use an LLM planner. Instead it
        records the failed execution and asks the existing
        controller for another decision.

        Phase 5 will replace this with model-driven planning.
        """

        state.metadata[
            "last_replan_reason"
        ] = (
            "Tool execution failure after "
            "retry limit was reached."
        )

        state.metadata[
            "replan_attempt"
        ] = state.replan_count

        # Keep the current step active. The controller will
        # determine whether another tool/action is possible.
        current_step = (
            state.current_step()
        )

        if current_step is not None:
            current_step.status = (
                "pending"
            )

    def _stop_for_limit(
        self,
        state: AgentState,
    ) -> None:
        state.add_error(
            (
                "Agent stopped because an execution "
                "limit was reached."
            )
        )

        state.metadata[
            "stopped_reason"
        ] = "execution_limit"

        state.set_status(
            AgentStatus.FAILED
        )

    @staticmethod
    def final_answer(
        state: AgentState,
    ) -> Optional[str]:
        return state.metadata.get(
            "final_answer"
        )