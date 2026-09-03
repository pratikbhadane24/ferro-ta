## Summary

<!-- Brief description of what this PR does. -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature / new indicator (non-breaking change)
- [ ] Breaking change (fix or feature that changes existing behaviour)
- [ ] Documentation / infrastructure only
- [ ] Language binding (wrapper only — no new algorithm)

## Checklist

- [ ] All existing tests pass (`pytest tests/`)
- [ ] New tests have been added for any new behaviour
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy --release -- -D warnings` passes
- [ ] `TA_LIB_COMPATIBILITY.md` accuracy table updated (if indicators were added or changed)
- [ ] CHANGELOG.md updated (for user-visible changes)
- [ ] Docstrings added or updated for new/changed public functions
- [ ] Indicator compute landed in `ferro_ta_core` first (bindings only wrap core)
- [ ] WASM / Flutter wrappers updated or regenerate skipped with a reason
- [ ] `python3 scripts/build_api_manifest.py` refreshed if public names changed
  (`docs/api_manifest.json` + `docs/languages/_coverage.inc.rst`)

## New language binding

Skip this section unless the PR adds a new language. New languages **must**
wrap `ferro_ta_core` — reimplementation is out of scope. See
`docs/languages/adding.rst`.

- [ ] Every compute call is `ferro_ta_core::…` (no ported loops)
- [ ] Binding style documented (direct FFI / WASM interop / generated wrappers)
- [ ] Repo layout, README, license, and package manifest
- [ ] API shape documented (names, buffers, NaN warmup, multi-output, errors)
- [ ] Coverage plan vs `docs/languages/coverage.rst` (`MANUAL_EXCLUDE`-style skips listed)
- [ ] Numeric tests: SMA, EMA, RSI, MACD, BBANDS, ATR, one CDL
- [ ] `ci-<lang>.yml` wired into `CI.yml`; generated wrappers `--check` if applicable
- [ ] Publish row in `RELEASE.md` + version carrier in `scripts/bump_version.py`
- [ ] Language page, README install row, support matrix, changelog
- [ ] `docs/rust_first.md` and CONTRIBUTING updated for the new binding

## Related Issues

Closes #<!-- issue number -->
