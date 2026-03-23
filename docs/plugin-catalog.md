# Plugin Catalog

A curated list of ferro-ta plugins and community extensions.

> **Note:** This catalog is community-maintained and provided on a best-effort
> basis.  Plugins are not endorsed or audited by the ferro-ta maintainers.
> Verify each plugin before use in production.

---

## How to Add Your Plugin

1. Verify that your plugin works with the current ferro-ta release.
2. Open a pull request adding a row to the table below.  Include:
   - **Name**: package name (PyPI or GitHub)
   - **Description**: one-line summary of what the plugin adds
   - **Install**: `pip install ...` command
   - **Link**: GitHub or PyPI URL

**Listing criteria:**
- Has a README or documentation describing what it does.
- Works with the current or previous minor version of ferro-ta.
- Is publicly available (PyPI or GitHub).

---

## Known Plugins

| Name | Description | Install | Link |
|------|-------------|---------|------|
| *(none yet — be the first!)* | | | |

---

## Reference Implementation

The `examples/custom_indicator.py` file in the ferro-ta repository serves as
the canonical reference for building a plugin.  See [Writing a plugin](plugins.rst)
for the full guide.

```python
# Minimal plugin example
from ferro_ta.registry import register
from ferro_ta import RSI, SMA

def SMOOTH_RSI(close, timeperiod=14, smooth=3):
    """Smoothed RSI: RSI of RSI values."""
    return SMA(RSI(close, timeperiod=timeperiod), timeperiod=smooth)

register("SMOOTH_RSI", SMOOTH_RSI)
```

---

## Publishing Your Plugin to PyPI

1. **Implement** your indicator(s) following the [plugin contract](plugins.rst).
2. **Package** with pyproject.toml using the `ferro_ta.plugins` entry point:

```toml
[project]
name = "ferro-ta-myplugin"
version = "1.0.0"
dependencies = ["ferro_ta>=1.0.0"]

[project.entry-points."ferro_ta.plugins"]
auto_register = "ferro_ta_myplugin:register_all"
```

3. **Publish** to PyPI:

```bash
pip install build twine
python -m build
twine upload dist/*
```

4. **Submit a PR** to add your plugin to this catalog.

---

## Removal Requests

If you are the maintainer of a listed plugin and want it removed, open a
GitHub issue with the title "Plugin catalog removal: <name>".
