# GeoGrapher Architecture Overview

GeoGrapher is organized as a reusable Python package with layered responsibilities that map directly to the specification.

## Package Layout

```
geographer/
  core/
    points.py        # Point primitives and segment utilities
    lines.py         # Line representations, conversions, intersections
    triangles.py     # Triangle centers, metrics, and derived constructs
    circles.py       # Circle algebra, intersections, tangents, power
    conics.py        # Parabola, ellipse, hyperbola helpers and classification
    matrices.py      # Linear algebra utilities and geometric transforms
    trig.py          # Trigonometric identity evaluators and graph helpers
    calculus.py      # Function parsing, calculus operations, critical point analysis
    locus.py         # Generic locus solvers and sampling helpers
    transforms.py    # Coordinate transform pipelines, homogeneous matrices
    utils.py         # Shared numeric helpers, tolerance handling, SymPy parsing
  viz/
    plotly_plotter.py  # Convert plot element specs into Plotly figures & layers
    exporter.py        # Export Plotly figures to PNG/SVG/PDF and JSON scene state
  ui/
    streamlit_app.py   # Streamlit front-end orchestrating the user workflow
  tests/
    test_*.py          # Pytest-based coverage for computational modules
  examples/
    notebooks/         # Scenario notebooks mirroring NCERT problem sets
```

## Layered Responsibilities

1. **Core**: Deterministic math logic with no UI dependencies. Inputs/outputs are plain dataclasses and JSON-friendly primitives. SymPy handles symbolic outputs while NumPy provides efficient numerics.
2. **Visualization**: Converts abstract plot element specifications (points, segments, loci, annotations) into concrete Plotly traces. Handles styling, themes, and export.
3. **UI**: Streamlit interface collects user input, invokes core computations, displays plots/derivations, persists history, and orchestrates exports.

## Data Contracts

- Core functions return `ComputationResult` objects containing:
  - `status`: `"ok" | "warning" | "error"`
  - `payload`: Numeric values, symbolic forms, or structures ready for display/export.
  - `work`: Ordered strings (plain text or LaTeX) describing the derivation.
  - `plot_elements`: Declarative specs like `{ "type": "segment", "points": ["A","B"], "style": { ... } }`.
  - `warnings`: Optional structured warnings (e.g., degeneracy, precision fallback).

- Visualization layer accepts `plot_elements` plus layout config and returns a `Plotly` figure.
- UI binds controls to specific `op` handlers defined in a routing table (see `ui/registry.py`).

## Extensibility Strategy

- Each math domain (e.g., `lines`, `circles`, `calculus`) exposes a registry of operations including metadata (required args, defaults, presets, plot behaviour). This enables the UI to dynamically render appropriate input widgets.
- Preset examples and random generators live beside their respective modules to ensure reproducibility and unit testing.
- History and export state rely on a canonical JSON schema stored in `geographer/schemas.py` so that CLI/notebook integrations share the same format.

## Testing Approach

- Unit tests target individual operations with known analytical results (e.g., NCERT answers) to verify numeric accuracy and symbolic derivations.
- Integration tests assemble end-to-end flows (e.g., computing triangle centers and verifying Plotly spec contents).
- Property-based tests (via `hypothesis`, optional) will ensure invariants like distance symmetry or determinant multiplicativity.

## Future Enhancements

- Animation helpers for calculus (Riemann sums) and parametric sweeps using Plotly frames.
- CLI entry point for scripted computations.
- Optional GPU acceleration for dense matrix operations via CuPy when available.
