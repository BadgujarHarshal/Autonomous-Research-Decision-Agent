# app/tools/calculator.py

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict

from app.tools.base import BaseTool


class SafeCalculatorTool(BaseTool):
    name = "calculate"
    description = (
        "Safely evaluates arithmetic expressions involving numbers, "
        "basic mathematical operators, and selected math functions."
    )

    _binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    _unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    _functions = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
    }

    @property
    def argument_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        self.validate_arguments(arguments)

        expression = arguments.get("expression")

        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a non-empty string.")

        if len(expression) > 500:
            raise ValueError("expression is too long.")

        tree = ast.parse(expression.strip(), mode="eval")

        result = self._evaluate(tree.body)

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Calculation produced a non-finite result.")

        return result

    def _evaluate(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError("Boolean values are not allowed.")

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Only numeric constants are allowed.")

        if isinstance(node, ast.BinOp):
            operation = self._binary_operators.get(type(node.op))

            if operation is None:
                raise ValueError(
                    f"Operator {type(node.op).__name__} is not allowed."
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = self._unary_operators.get(type(node.op))

            if operation is None:
                raise ValueError(
                    f"Unary operator {type(node.op).__name__} is not allowed."
                )

            return operation(self._evaluate(node.operand))

        if isinstance(node, ast.Name):
            constants = {
                "pi": math.pi,
                "e": math.e,
            }

            if node.id in constants:
                return constants[node.id]

            raise ValueError(f"Unknown name: {node.id}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct math functions are allowed.")

            function = self._functions.get(node.func.id)

            if function is None:
                raise ValueError(
                    f"Function '{node.func.id}' is not allowed."
                )

            if node.keywords:
                raise ValueError("Keyword arguments are not allowed.")

            values = [
                self._evaluate(argument)
                for argument in node.args
            ]

            return function(*values)

        raise ValueError(
            f"Expression element '{type(node).__name__}' is not allowed."
        )