try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from . import config, ops  # noqa: F401
from .para_attn_interface import ring_attn_func, RingAttnMode, ulysses_attn_func, UlyssesAttnMode, UnifiedAttnMode

__all__ = [
    "UnifiedAttnMode",
    "RingAttnMode",
    "UlyssesAttnMode",
    "ring_attn_func",
    "ulysses_attn_func",
]
