"DACP solver"

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .solver import eigvalsh, estimated_errors

__all__ = [
    "eigvalsh",
    "estimated_errors",
    "__version__",
    "__version_tuple__",
]
