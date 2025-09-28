"""Tests for triangle metrics."""

from geographer.core.points import Point
from geographer.core.triangles import TriangleMetrics


def test_triangle_summary_key_metrics():
    triangle = TriangleMetrics(Point(0, 0, "A"), Point(4, 0, "B"), Point(0, 3, "C"))
    result = triangle.summary()
    payload = result.payload
    assert abs(payload["area"] - 6.0) < 1e-9
    centroid = payload["centroid"]
    assert abs(centroid[0] - (4.0 / 3.0)) < 1e-9
    assert abs(centroid[1] - 1.0) < 1e-9
    assert payload["classification"]["right"]
