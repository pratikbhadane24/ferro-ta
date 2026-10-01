"""Unit tests for the C-ABI code generators (scripts/ffi_codegen)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ffi_codegen.go import (  # noqa: E402
    check_unique_idents,
    go_enum_const,
    go_ident,
    go_name,
    go_out_name,
    go_requires,
)


def test_go_name_is_camel_case_of_snake() -> None:
    assert go_name("linearreg_slope") == "LinearregSlope"
    assert go_name("cdl3blackcrows") == "Cdl3blackcrows"


def test_go_ident_escapes_keywords_and_generated_locals() -> None:
    assert go_ident("type") == "type_"
    assert go_ident("range") == "range_"
    assert go_ident("len") == "len_"
    assert go_ident("err") == "err_"
    assert go_ident("n") == "n_"


def test_go_ident_keeps_harmless_builtin_shadowing() -> None:
    # Shadowing close/real is legal and keeps public signatures readable.
    assert go_ident("close") == "close"
    assert go_ident("fastk_period") == "fastkPeriod"


def test_go_out_name_strips_prefix() -> None:
    assert go_out_name("out") == "out"
    assert go_out_name("out_upper") == "upper"
    assert go_out_name("out_type") == "type_"


def test_duplicate_identifiers_fail_with_spec_name() -> None:
    with pytest.raises(ValueError, match="ft_bad"):
        check_unique_idents("ft_bad", ["upper", "lower", "upper"])
    check_unique_idents("ft_ok", ["upper", "lower"])


def test_requires_rule_uses_go_parameter_names() -> None:
    fn = {
        "doc": "Doc.",
        "requires": "long_window > short_window",
        "params": [{"name": "short_window"}, {"name": "long_window"}],
    }
    assert (
        go_requires(fn)
        == "Doc.\nRequires: longWindow > shortWindow (else ErrInvalidParam)."
    )
    assert go_requires({"doc": "Doc.", "requires": "", "params": []}) == "Doc."


def test_enum_constant_names() -> None:
    assert go_enum_const("FT_OPTION", "CALL") == "OptionCall"
    assert go_enum_const("FT_MODEL", "BLACK_76") == "ModelBlack76"
