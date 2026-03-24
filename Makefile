# ferro-ta development Makefile
# Usage: make <target>

.PHONY: help dev build test lint typecheck fmt docs clean bench version audit prepush hooks

# Default target
help:
	@echo "ferro-ta development targets:"
	@echo ""
	@echo "  make dev        Install dev dependencies (maturin + test extras)"
	@echo "  make build      Build and install the Rust extension in dev mode"
	@echo "  make test       Run the full Python test suite with coverage"
	@echo "  make lint       Run ruff linter on python/ and tests/"
	@echo "  make fmt        Run rustfmt + ruff formatter"
	@echo "  make typecheck  Run mypy + pyright type checkers"
	@echo "  make docs       Build the Sphinx documentation"
	@echo "  make bench      Run Rust criterion benchmarks (ferro_ta_core)"
	@echo "  make version    Bump tracked version strings (set VERSION=X.Y.Z)"
	@echo "  make audit      Run cargo-audit + pip-audit"
	@echo "  make prepush    Run the local pre-push CI gate (set CHECKS='version rust_fmt' to scope it)"
	@echo "  make hooks      Install pre-commit and pre-push git hooks"
	@echo "  make clean      Remove build artefacts"

dev:
	pip install uv
	uv pip install --system maturin numpy pytest pytest-cov pandas polars hypothesis pyyaml \
	    sphinx sphinx-rtd-theme ruff mypy pyright pre-commit

build:
	maturin develop --release

test: build
	pytest tests/ -v --cov=ferro_ta --cov-report=term-missing --cov-fail-under=65

lint:
	uv run --with ruff ruff check python/ tests/
	uv run --with ruff ruff format --check python/ tests/

fmt:
	cargo fmt --all
	uv run --with ruff ruff format python/ tests/

typecheck:
	uv run --with mypy --with numpy mypy python/ferro_ta --ignore-missing-imports --no-error-summary
	uv run --with pyright pyright python/ferro_ta

docs:
	pip install sphinx sphinx-rtd-theme
	sphinx-build -b html docs docs/_build --keep-going

bench:
	cargo bench -p ferro_ta_core

version:
	@test -n "$(VERSION)" || (echo "Usage: make version VERSION=X.Y.Z" && exit 1)
	python3 scripts/bump_version.py "$(VERSION)"

audit:
	cargo audit
	uv run --with pip-audit pip-audit

prepush:
	bash scripts/pre_push_checks.sh $(CHECKS)

hooks:
	uv run --with pre-commit pre-commit install --hook-type pre-commit --hook-type pre-push

clean:
	cargo clean
	rm -rf dist/ docs/_build/ coverage.xml .coverage *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.so" -delete 2>/dev/null || true
