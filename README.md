# GeoGrapher — Interactive Coordinate Geometry + Math Visualizer

GeoGrapher is a Streamlit-powered visual laboratory for NCERT Class 10–12 mathematics. It couples symbolic derivations powered by SymPy with interactive Plotly graphics so that each computation is visualized instantly.

## Highlights

- **Coordinate Geometry suite**: points, lines, triangles, circles, conics, loci, and transformations.
- **Linear algebra insights**: determinants, inverses, eigenvectors, and geometric action of matrices on shapes.
- **Trigonometry playground**: manipulate amplitude/frequency/phase, verify identities, and solve equations.
- **Calculus explorer**: plot functions, tangents, definite integrals, and Taylor approximations.
- **Reusable core API**: deterministic computational modules with structured JSON outputs.
- **Interactive UI**: Streamlit layout with sidebar inputs, Plotly canvas, LaTeX derivation steps, and export tools.

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app

```bash
streamlit run geographer/ui/streamlit_app.py
```

The app opens in your browser with dedicated modes for coordinate geometry, matrices, trigonometry, and calculus.

### 3. Run tests

```bash
pytest geographer/tests
```

## Repository Layout

```
geographer/
  core/            # Computational modules (points, lines, triangles, circles, conics, matrices, trig, calculus, transforms, loci)
  viz/             # Plotly trace builders and export helpers
  ui/              # Streamlit application
  tests/           # Pytest suite covering core modules
  examples/        # Starter notebooks with NCERT-style problems
```

Additional architectural details live in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Example Usage

Programmatic usage returns structured results ready for JSON export or plotting:

```python
from geographer.core import Point, distance

result = distance(Point(1, 2), Point(4, 6))
print(result.payload["distance"])  # 5.0
print(result.steps)                 # LaTeX-friendly derivation strings
print(result.plot_elements)         # Declarative plot specification
```

## Roadmap

- Enrich preset NCERT examples and add random problem generators.
- Extend conics, loci, and transformation coverage with additional edge-case handling.
- Add calculus animations (Riemann sums, Newton iterations) and export-ready narratives.
- Package CLI and Docker image for reproducible deployments.

## License

GeoGrapher is released under the MIT license.
