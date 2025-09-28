"""Tests for point operations."""

from geographer.core.points import Point, distance, midpoint, section_point


def test_distance_between_points():
    result = distance(Point(1, 2, "A"), Point(4, 6, "B"))
    assert abs(result.payload["distance"] - 5.0) < 1e-9


def test_midpoint():
    result = midpoint(Point(0, 0, "A"), Point(4, 0, "B"))
    assert result.payload["midpoint"]["x"] == 2
    assert result.payload["midpoint"]["y"] == 0


def test_section_point_internal():
    result = section_point(Point(0, 0), Point(4, 0), (1, 3))
    point = result.payload["point"]
    assert abs(point["x"] - 3) < 1e-9
    assert abs(point["y"]) < 1e-9
