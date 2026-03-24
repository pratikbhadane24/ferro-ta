#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AVAILABLE_CHECKS=(
  version
  changelog
  rust_fmt
  rust_clippy
  rust_core
  rust_bench
  python_lint
  python_typecheck
  python_test
  docs
  wasm
  manifest
)

DEFAULT_CHECKS=("${AVAILABLE_CHECKS[@]}")

python_env_ready=0

usage() {
  cat <<'EOF'
Usage:
  scripts/pre_push_checks.sh
  scripts/pre_push_checks.sh <check> [<check> ...]
  scripts/pre_push_checks.sh --list

Runs the repo's basic local CI gate before push. By default it covers:
  version changelog rust_fmt rust_clippy rust_core rust_bench
  python_lint python_typecheck python_test docs wasm manifest

Notes:
  - This mirrors the required CI categories we can run locally.
  - It intentionally skips the multi-Python test matrix, audit jobs, perf smoke,
    and benchmark-regression jobs.
EOF
}

list_checks() {
  printf '%s\n' "${AVAILABLE_CHECKS[@]}"
}

need_cmd() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

run_cmd() {
  printf '  +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

ensure_python_env() {
  if [[ "$python_env_ready" -eq 1 ]]; then
    return
  fi

  need_cmd uv
  run_cmd uv sync --extra dev --extra docs
  run_cmd uv run --extra dev --extra docs maturin develop --release
  python_env_ready=1
}

run_version() {
  need_cmd python3
  run_cmd python3 scripts/bump_version.py --check
}

run_changelog() {
  need_cmd python3
  run_cmd python3 scripts/check_changelog.py
}

run_rust_fmt() {
  need_cmd cargo
  run_cmd cargo fmt --all -- --check
}

run_rust_clippy() {
  need_cmd cargo
  run_cmd cargo clippy --release -- -D warnings
}

run_rust_core() {
  need_cmd cargo
  run_cmd cargo build -p ferro_ta_core
  run_cmd cargo test -p ferro_ta_core
}

run_rust_bench() {
  need_cmd cargo
  run_cmd cargo bench -p ferro_ta_core --no-run
}

run_python_lint() {
  need_cmd uv
  run_cmd uv run --with ruff ruff check python/ tests/
  run_cmd uv run --with ruff ruff format --check python/ tests/
}

run_python_typecheck() {
  need_cmd uv
  run_cmd uv run --with mypy --with numpy mypy python/ferro_ta --ignore-missing-imports --no-error-summary
  run_cmd uv run --with pyright pyright python/ferro_ta
}

run_python_test() {
  ensure_python_env
  run_cmd uv run --extra dev --with pytest-cov pytest tests/unit/ tests/integration/ -v --cov=ferro_ta --cov-report=term-missing --cov-fail-under=65
}

run_docs() {
  ensure_python_env
  run_cmd uv run --extra docs python -m sphinx -b html docs docs/_build -W --keep-going
}

run_wasm() {
  need_cmd node
  need_cmd wasm-pack
  local benchmark_json="../.wasm_benchmark.prepush.json"
  (
    cd wasm
    trap 'rm -f "$benchmark_json"' EXIT
    run_cmd wasm-pack test --node
    run_cmd wasm-pack build --target nodejs --out-dir pkg
    run_cmd node bench.js --json "$benchmark_json"
  )
}

run_manifest() {
  need_cmd python3
  run_cmd python3 scripts/check_api_manifest.py
}

run_check() {
  local check_name="$1"
  case "$check_name" in
    version) run_version ;;
    changelog) run_changelog ;;
    rust_fmt) run_rust_fmt ;;
    rust_clippy) run_rust_clippy ;;
    rust_core) run_rust_core ;;
    rust_bench) run_rust_bench ;;
    python_lint) run_python_lint ;;
    python_typecheck) run_python_typecheck ;;
    python_test) run_python_test ;;
    docs) run_docs ;;
    wasm) run_wasm ;;
    manifest) run_manifest ;;
    *)
      echo "Unknown check: $check_name" >&2
      echo "Use --list to see supported checks." >&2
      exit 1
      ;;
  esac
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
  list_checks
  exit 0
fi

selected_checks=()
if [[ "$#" -gt 0 ]]; then
  selected_checks=("$@")
else
  selected_checks=("${DEFAULT_CHECKS[@]}")
fi

total_checks="${#selected_checks[@]}"
index=0
for check_name in "${selected_checks[@]}"; do
  index=$((index + 1))
  printf '\n[%d/%d] %s\n' "$index" "$total_checks" "$check_name"
  run_check "$check_name"
done

printf '\nAll selected pre-push checks passed.\n'
