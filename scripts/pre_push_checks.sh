#!/usr/bin/env bash
# Pre-push CI gate — runs checks in parallel to minimise wall-clock time.
#
# Usage:
#   scripts/pre_push_checks.sh                    # all checks
#   scripts/pre_push_checks.sh rust_clippy wasm   # selected checks
#   scripts/pre_push_checks.sh --list
#   FERRO_FAST=1 scripts/pre_push_checks.sh       # skip docs + wasm bench
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AVAILABLE_CHECKS=(
  version changelog manifest
  rust_fmt rust_clippy rust_core rust_bench
  python_lint python_typecheck python_test
  docs wasm
)
DEFAULT_CHECKS=("${AVAILABLE_CHECKS[@]}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2; exit 1
  fi
}

run_cmd() {
  printf '    +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/pre_push_checks.sh
  scripts/pre_push_checks.sh <check> [<check> ...]
  scripts/pre_push_checks.sh --list

Environment:
  FERRO_FAST=1   Skip docs and wasm (fastest local feedback loop)
EOF
}

# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

run_version()     { need_cmd python3; run_cmd python3 scripts/bump_version.py --check; }
run_changelog()   { need_cmd python3; run_cmd python3 scripts/check_changelog.py; }
run_manifest()    { need_cmd python3; run_cmd python3 scripts/check_api_manifest.py; }
run_rust_fmt()    { need_cmd cargo;   run_cmd cargo fmt --all -- --check; }
run_python_lint() {
  need_cmd uv
  run_cmd uv run --with ruff ruff check python/ tests/
  run_cmd uv run --with ruff ruff format --check python/ tests/
}

run_rust_clippy()      { need_cmd cargo; run_cmd cargo clippy --release -- -D warnings; }
run_rust_core()        { need_cmd cargo; run_cmd cargo build -p ferro_ta_core && run_cmd cargo test -p ferro_ta_core; }
run_rust_bench()       { need_cmd cargo; run_cmd cargo bench -p ferro_ta_core --no-run; }

run_python_typecheck() {
  need_cmd uv
  run_cmd uv run --with mypy --with numpy python -m mypy python/ferro_ta \
    --ignore-missing-imports --no-error-summary
  run_cmd uv run --with pyright python -m pyright python/ferro_ta
}

# python_test and docs both need a compiled extension.
# Use a flag file so only the first concurrent caller runs maturin develop;
# subsequent callers (in parallel background jobs) wait and reuse it.
_MATURIN_LOCK="${TMPDIR:-/tmp}/ferro_ta_maturin_$$.lock"
_MATURIN_FLAG="${TMPDIR:-/tmp}/ferro_ta_maturin_$$.done"

ensure_python_env() {
  [[ -f "$_MATURIN_FLAG" ]] && return
  (
    flock 9
    if [[ ! -f "$_MATURIN_FLAG" ]]; then
      need_cmd uv
      run_cmd uv sync --extra dev --extra docs --extra mcp
      run_cmd uv run --extra dev --extra docs --extra mcp maturin develop --release
      touch "$_MATURIN_FLAG"
    fi
  ) 9>"$_MATURIN_LOCK"
}

run_python_test() {
  ensure_python_env
  run_cmd uv run --extra dev --extra mcp --with pytest-cov \
    pytest tests/unit/ tests/integration/ \
    -v --cov=ferro_ta --cov-report=term-missing --cov-fail-under=65
}

run_docs() {
  ensure_python_env
  run_cmd uv run --extra docs python -m sphinx -b html docs docs/_build -W --keep-going
}

run_wasm() {
  need_cmd node; need_cmd wasm-pack
  (
    cd wasm
    run_cmd wasm-pack test --node
    run_cmd npm run build
    if [[ "${FERRO_FAST:-0}" != "1" ]]; then
      local bj="../.wasm_benchmark.prepush.json"
      run_cmd node bench.js --json "$bj"
      rm -f "$bj"
    fi
  )
}

run_check() {
  case "$1" in
    version)          run_version ;;
    changelog)        run_changelog ;;
    manifest)         run_manifest ;;
    rust_fmt)         run_rust_fmt ;;
    rust_clippy)      run_rust_clippy ;;
    rust_core)        run_rust_core ;;
    rust_bench)       run_rust_bench ;;
    python_lint)      run_python_lint ;;
    python_typecheck) run_python_typecheck ;;
    python_test)      run_python_test ;;
    docs)             run_docs ;;
    wasm)             run_wasm ;;
    *) echo "Unknown check: $1 — use --list" >&2; exit 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Parallel runner — starts all checks concurrently, collects results
# ---------------------------------------------------------------------------

run_parallel() {
  local -a checks=("$@")
  [[ "${#checks[@]}" -eq 0 ]] && return 0

  local -a pids logs names
  local start
  start=$(date +%s)

  printf '\nStarting %d checks in parallel: %s\n' "${#checks[@]}" "${checks[*]}"

  for check in "${checks[@]}"; do
    local log
    log=$(mktemp /tmp/ferro_prepush_XXXXXX)
    logs+=("$log")
    names+=("$check")
    run_check "$check" >"$log" 2>&1 &
    pids+=($!)
  done

  local failed=0
  local -a failed_names
  printf '\n'
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}" 2>/dev/null; then
      printf '  ✓ %s\n' "${names[$i]}"
    else
      printf '  ✗ %s\n' "${names[$i]}"
      failed_names+=("${names[$i]}")
      failed=1
    fi
  done

  # Print logs for failed checks only
  if [[ "$failed" -eq 1 ]]; then
    for i in "${!names[@]}"; do
      local name="${names[$i]}"
      if [[ " ${failed_names[*]:-} " == *" $name "* ]]; then
        printf '\n'; printf '━%.0s' {1..60}; printf '\nFAILED: %s\n' "$name"; printf '━%.0s' {1..60}; printf '\n'
        cat "${logs[$i]}"
      fi
    done
  fi

  for log in "${logs[@]}"; do rm -f "$log"; done
  rm -f "$_MATURIN_LOCK" "$_MATURIN_FLAG"

  printf '\nElapsed: %ds\n' "$(( $(date +%s) - start ))"
  return "$failed"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

[[ "${1:-}" == "--help" || "${1:-}" == "-h" ]] && { usage; exit 0; }
[[ "${1:-}" == "--list" ]] && { printf '%s\n' "${AVAILABLE_CHECKS[@]}"; exit 0; }

selected_checks=()
if [[ "$#" -gt 0 ]]; then
  selected_checks=("$@")
else
  selected_checks=("${DEFAULT_CHECKS[@]}")
  if [[ "${FERRO_FAST:-0}" == "1" ]]; then
    selected_checks=()
    for c in "${DEFAULT_CHECKS[@]}"; do
      [[ "$c" == "docs" || "$c" == "wasm" ]] && continue
      selected_checks+=("$c")
    done
    printf 'FERRO_FAST=1: skipping docs + wasm\n'
  fi
fi

# ---------------------------------------------------------------------------
# Execution strategy:
#   Phase 1 — instant gate (sequential, fail-fast):
#              version, changelog, manifest, python_lint, rust_fmt
#              These are trivial to run and catch the most common mistakes early.
#              If any fail here we abort immediately without waiting for slow checks.
#
#   Phase 2 — everything else in parallel:
#              rust_clippy, rust_core, rust_bench, python_typecheck,
#              python_test, docs, wasm
# ---------------------------------------------------------------------------

FAST_CHECKS=(version changelog manifest python_lint rust_fmt)

phase1=()
phase2=()
for c in "${selected_checks[@]}"; do
  is_fast=0
  for f in "${FAST_CHECKS[@]}"; do [[ "$c" == "$f" ]] && is_fast=1 && break; done
  if [[ "$is_fast" -eq 1 ]]; then phase1+=("$c"); else phase2+=("$c"); fi
done

# Phase 1: fast gate
if [[ "${#phase1[@]}" -gt 0 ]]; then
  printf 'Phase 1 — fast gate (%d checks)\n' "${#phase1[@]}"
  start1=$(date +%s)
  for c in "${phase1[@]}"; do
    printf '  [%s] ... ' "$c"
    log=$(mktemp /tmp/ferro_prepush_XXXXXX)
    if run_check "$c" >"$log" 2>&1; then
      printf 'ok\n'
    else
      printf 'FAILED\n'
      cat "$log"
      rm -f "$log"
      echo "" >&2
      echo "Fast gate failed on '$c' — aborting before slow checks." >&2
      exit 1
    fi
    rm -f "$log"
  done
  printf 'Phase 1 passed (%ds)\n' "$(( $(date +%s) - start1 ))"
fi

# Phase 2: parallel slow checks
if [[ "${#phase2[@]}" -gt 0 ]]; then
  printf '\nPhase 2 — parallel slow checks\n'
  run_parallel "${phase2[@]}" || exit 1
fi

printf '\nAll pre-push checks passed.\n'
