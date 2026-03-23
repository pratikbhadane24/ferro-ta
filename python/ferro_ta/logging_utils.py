"""Backward-compat stub — moved to ``ferro_ta.core.logging_utils``."""

from ferro_ta.core.logging_utils import *  # noqa: F401, F403

try:
    from ferro_ta.core.logging_utils import __all__  # noqa: F401
except ImportError:
    pass
