"""Tests for line operations."""

from geographer.core.lines import (
    Line,
    angle_between_lines,
    intersection,
    line_from_points,
)
from geographer.core.points import Point


def test_line_from_points_general_form():
    result = line_from_points(Point(0, 0, "A"), Point(2, 2, "B"))
    line_payload = result.payload["line"]
    # For y = x, A = -1, B = 1 (normalized, but we only check slope)
    assert abs(line_payload["slope"] - 1.0) < 1e-9


def test_intersection_of_lines():
    line1 = Line.from_general(1, -1, 0)
    line2 = Line.from_general(0, 1, -2)
    result = intersection(line1, line2)
    x, y = result.payload["point"]
    assert abs(x - 2.0) < 1e-9
    assert abs(y - 2.0) < 1e-9


def test_angle_between_perpendicular_lines():
    line1 = Line.from_general(1, 0, 0)
    line2 = Line.from_general(0, 1, 0)
    result = angle_between_lines(line1, line2)
    assert abs(result.payload["angle_degrees"] - 90.0) < 1e-6
