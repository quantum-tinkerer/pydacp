"DACP solver"

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .solver import eigvalsh

__all__ = [
    "eigvalsh",
    "__version__",
    "__version_tuple__",
]
