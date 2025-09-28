"""GeoGrapher core package entry point."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("geographer")
except PackageNotFoundError:  # pragma: no cover - during local development
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
