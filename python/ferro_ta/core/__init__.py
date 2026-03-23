"""
ferro_ta.core — Core utilities: exceptions, configuration, logging, registry, raw bindings.

Sub-modules
-----------
* :mod:`ferro_ta.core.exceptions`    — Custom exception hierarchy and error helpers
* :mod:`ferro_ta.core.config`        — Global configuration and defaults
* :mod:`ferro_ta.core.logging_utils` — Debug-logging helpers
* :mod:`ferro_ta.core.registry`      — Indicator function registry
* :mod:`ferro_ta.core.raw`           — Raw Rust-binding wrappers (zero-overhead pass-through)

Import directly from sub-modules to avoid circular dependencies, e.g.::

    from ferro_ta.core.exceptions import FerroTAError
    from ferro_ta.core.registry import register, run
"""
