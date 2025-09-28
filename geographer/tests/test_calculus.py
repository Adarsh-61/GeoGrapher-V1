"""Tests for calculus utilities."""

from geographer.core.calculus import FunctionAnalyzer


def test_function_analyzer_derivative():
    analyzer = FunctionAnalyzer("x**3")
    result = analyzer.derivative_at(2)
    assert abs(result.payload["slope"] - 12.0) < 1e-9


def test_definite_integral():
    analyzer = FunctionAnalyzer("x")
    result = analyzer.definite_integral(0, 2)
    assert abs(result.payload["integral"] - 2.0) < 1e-9
