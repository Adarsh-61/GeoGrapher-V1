"""Core analytic modules for GeoGrapher."""

# Re-export commonly used helpers for convenience
from .points import (
    Point,
    distance,
    midpoint,
    parametric_point,
    section_point,
)
from .lines import (
    Line,
    angle_between_lines,
    distance_from_point,
    foot_of_perpendicular,
    intersection,
    line_from_points,
)
from .triangles import TriangleMetrics
from .circles import (
    Circle,
    circle_circle_intersection,
    line_circle_intersection,
    radical_axis,
    tangents_from_external_point,
)
from .conics import (
    classify_general_quadratic,
    ellipse_standard,
    hyperbola_standard,
    line_conic_intersection,
    parabola_x2_4ay,
    parabola_y2_4ax,
)
from .matrices import (
    eigen_analysis,
    matrix_addition,
    matrix_determinant,
    matrix_inverse,
    matrix_multiplication,
    matrix_transform,
)
from .trig import (
    plot_basic_trig,
    solve_trig_equation,
    transform_trig,
    verify_identity,
)
from .calculus import FunctionAnalyzer
from .locus import apollonius_circle, locus_midpoints, perpendicular_bisector
from .transforms import (
    apply_affine,
    apply_reflection,
    apply_rotation,
    apply_scaling,
    apply_translation,
)

__all__ = [
    "Point",
    "distance",
    "midpoint",
    "section_point",
    "parametric_point",
    "Line",
    "intersection",
    "line_from_points",
    "angle_between_lines",
    "distance_from_point",
    "foot_of_perpendicular",
    "TriangleMetrics",
    "Circle",
    "line_circle_intersection",
    "circle_circle_intersection",
    "tangents_from_external_point",
    "radical_axis",
    "parabola_y2_4ax",
    "parabola_x2_4ay",
    "ellipse_standard",
    "hyperbola_standard",
    "classify_general_quadratic",
    "line_conic_intersection",
    "matrix_addition",
    "matrix_multiplication",
    "matrix_determinant",
    "matrix_inverse",
    "matrix_transform",
    "eigen_analysis",
    "plot_basic_trig",
    "transform_trig",
    "verify_identity",
    "solve_trig_equation",
    "FunctionAnalyzer",
    "perpendicular_bisector",
    "apollonius_circle",
    "locus_midpoints",
    "apply_translation",
    "apply_rotation",
    "apply_scaling",
    "apply_reflection",
    "apply_affine",
]
