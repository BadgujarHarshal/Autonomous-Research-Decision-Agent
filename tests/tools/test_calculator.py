#  tests/tools/test_calculator.py

import pytest

from app.tools.calculator import SafeCalculatorTool


@pytest.fixture
def calculator():
    return SafeCalculatorTool()


def test_addition(calculator):
    assert calculator.execute(
        {"expression": "10 + 20"}
    ) == 30


def test_operator_precedence(calculator):
    assert calculator.execute(
        {"expression": "10 + 20 * 2"}
    ) == 50


def test_division(calculator):
    assert calculator.execute(
        {"expression": "20 / 5"}
    ) == 4


def test_power(calculator):
    assert calculator.execute(
        {"expression": "2 ** 3"}
    ) == 8


def test_math_function(calculator):
    assert calculator.execute(
        {"expression": "sqrt(16)"}
    ) == 4


def test_pi(calculator):
    result = calculator.execute(
        {"expression": "pi * 2"}
    )

    assert abs(result - 6.283185307179586) < 1e-9


def test_empty_expression_rejected(calculator):
    with pytest.raises(ValueError):
        calculator.execute(
            {"expression": ""}
        )


def test_invalid_expression_rejected(calculator):
    with pytest.raises(ValueError):
        calculator.execute(
            {"expression": "unknown_function(10)"}
        )


def test_python_code_execution_is_blocked(calculator):
    with pytest.raises(ValueError):
        calculator.execute(
            {
                "expression": "__import__('os').system('dir')"
            }
        )


def test_attribute_access_is_blocked(calculator):
    with pytest.raises(ValueError):
        calculator.execute(
            {
                "expression": "math.sqrt(16)"
            }
        )


def test_boolean_is_blocked(calculator):
    with pytest.raises(ValueError):
        calculator.execute(
            {
                "expression": "True"
            }
        )