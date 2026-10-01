# ferro-ta development Makefile
# Usage: make <target>

.PHONY: help dev build test lint typecheck fmt docs clean bench version audit prepush hooks flutter flutter-gen ffi-gen go-lib go c-smoke

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
	@echo "  make flutter    Verify the Flutter binding (fresh wrappers + core parity)"
	@echo "  make flutter-gen Regenerate Flutter api wrappers + flutter_rust_bridge glue"
	@echo "  make ffi-gen    Regenerate ffi_spec.json, ferro_ta.h and the Go wrappers"
	@echo "  make go-lib     Build the C ABI static library into bindings/go/lib/<host>"
	@echo "  make go         Verify the Go binding (fresh wrappers, vet, race tests)"
	@echo "  make c-smoke    Compile + run the C smoke test against ferro_ta.h"
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

# Regenerate the Flutter api wrappers (from WASM signatures) and the
# flutter_rust_bridge Dart/Rust glue. Requires the Flutter SDK + FRB codegen.
flutter-gen:
	python3 scripts/build_flutter_bridge.py
	cd flutter && flutter_rust_bridge_codegen generate

# Verify the Flutter binding: generated wrappers are fresh and compile against
# the core crate (Dart/Flutter checks run when the SDK is installed).
flutter:
	python3 scripts/build_flutter_bridge.py --check
	cd flutter/rust && RUSTFLAGS="" cargo build && RUSTFLAGS="" cargo test

# Regenerate the C ABI spec, the C header and the Go wrappers.
ffi-gen:
	python3 scripts/build_ffi_bindings.py

# Build the C ABI static archive for this host into the Go module's lib/ dir
# (release tags ship these prebuilt; on main, lib/ is gitignored).
GO_HOST := $(shell go env GOOS 2>/dev/null)_$(shell go env GOARCH 2>/dev/null)
go-lib:
	cargo build -p ferro_ta_ffi --release
	mkdir -p bindings/go/lib/$(GO_HOST)
	cp target/release/libferro_ta_ffi.a bindings/go/lib/$(GO_HOST)/

# Verify the Go binding: generated code is fresh, then vet + race tests.
go: go-lib
	python3 scripts/build_ffi_bindings.py --check
	cd bindings/go && go vet ./... && go test -race ./...

# Prove the generated header compiles with strict warnings and links from C.
c-smoke:
	cargo build -p ferro_ta_ffi --release
	$(CC) -std=c99 -Wall -Wextra -Werror -pedantic -Icrates/ferro_ta_ffi/include \
		crates/ferro_ta_ffi/tests/c_smoke/main.c target/release/libferro_ta_ffi.a -lm \
		-o target/release/c_smoke
	./target/release/c_smoke

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
