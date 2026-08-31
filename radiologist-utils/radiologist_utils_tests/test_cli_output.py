# MIT License
#
# Copyright (c) 2026 @CedrickArmel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Behavioral tests for the shared CLI output/exit-code seam.

Drives ``radiologist.utils.cli``'s public API (``emit``, ``resolve_format``,
``exit_code_for``) — never the internal ``output``/``exits`` submodules.
"""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from radiologist.utils.cli import (
    EXIT_ERROR,
    EXIT_NOT_FOUND,
    emit,
    exit_code_for,
    resolve_format,
)

# ---------------------------------------------------------------------------
# emit() — kv mode (default)
# ---------------------------------------------------------------------------


def test_emit_writes_one_key_value_line_per_entry_in_insertion_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data: Dict[str, Any] = OrderedDict([("status", "ok"), ("count", 3)])
    emit(data)
    out = capsys.readouterr().out
    assert out == "status=ok\ncount=3\n"


def test_emit_kv_flattens_nested_mapping_with_dot_joined_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {"probabilities": OrderedDict([("NORMAL", 0.92), ("ABNORMAL", 0.08)])}
    emit(data)
    out = capsys.readouterr().out
    assert out == "probabilities.NORMAL=0.92\nprobabilities.ABNORMAL=0.08\n"


def test_emit_kv_indexes_list_values_with_brackets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {"aliases": ["best", "latest"]}
    emit(data)
    out = capsys.readouterr().out
    assert out == "aliases[0]=best\naliases[1]=latest\n"


def test_emit_kv_renders_none_value_as_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {"saliency_path": None}
    emit(data)
    out = capsys.readouterr().out
    assert out == "saliency_path=\n"


def test_emit_kv_renders_bool_values_lowercase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {"cached": True, "stale": False}
    emit(data)
    out = capsys.readouterr().out
    assert out == "cached=true\nstale=false\n"


def test_emit_kv_renders_empty_list_as_empty_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data: Dict[str, Any] = {"aliases": []}
    emit(data)
    out = capsys.readouterr().out
    assert out == "aliases=\n"


def test_emit_kv_renders_empty_mapping_as_empty_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data: Dict[str, Any] = {"probabilities": {}}
    emit(data)
    out = capsys.readouterr().out
    assert out == "probabilities=\n"


# ---------------------------------------------------------------------------
# emit() — json mode
# ---------------------------------------------------------------------------


def test_emit_json_mode_from_env_var_round_trips_with_nesting_preserved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
    data = {"probabilities": {"NORMAL": 0.92, "ABNORMAL": 0.08}}
    emit(data)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == data


def test_emit_json_mode_two_calls_produce_two_independently_valid_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit({"id": 1}, fmt="json")
    emit({"id": 2}, fmt="json")
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"id": 1}
    assert json.loads(lines[1]) == {"id": 2}


# ---------------------------------------------------------------------------
# emit() — yaml mode
# ---------------------------------------------------------------------------


def test_emit_yaml_mode_writes_document_with_explicit_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RADIOLOGIST_OUTPUT", "yaml")
    emit({"status": "ok"})
    out = capsys.readouterr().out
    assert out.startswith("---")


def test_emit_yaml_mode_two_calls_round_trip_as_two_documents(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit({"id": 1}, fmt="yaml")
    emit({"id": 2}, fmt="yaml")
    out = capsys.readouterr().out
    docs = list(yaml.safe_load_all(out))
    assert docs == [{"id": 1}, {"id": 2}]


# ---------------------------------------------------------------------------
# resolve_format()
# ---------------------------------------------------------------------------


def test_explicit_format_argument_takes_precedence_over_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RADIOLOGIST_OUTPUT", "yaml")
    assert resolve_format("json") == "json"


def test_resolve_format_raises_value_error_naming_offending_value() -> None:
    with pytest.raises(ValueError, match="xml"):
        resolve_format("xml")


def test_resolve_format_normalizes_whitespace_and_case() -> None:
    assert resolve_format("  JSON  ") == "json"


# ---------------------------------------------------------------------------
# emit() — yaml extra guard
# ---------------------------------------------------------------------------


def test_emit_yaml_mode_raises_runtime_error_when_pyyaml_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radiologist.utils.cli import output as output_module

    monkeypatch.setattr(output_module, "_yaml", None)
    with pytest.raises(RuntimeError, match=r"radiologist-utils\[cli\]"):
        emit({"status": "ok"}, fmt="yaml")


# ---------------------------------------------------------------------------
# emit() — numpy scalar normalization
# ---------------------------------------------------------------------------


def test_emit_kv_mode_renders_numpy_scalars_as_plain_numbers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {"score": np.float32(0.5)}
    emit(data, fmt="kv")
    out = capsys.readouterr().out
    assert out == "score=0.5\n"


def test_emit_json_mode_renders_numpy_scalars_without_raising(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {"score": np.float32(0.5)}
    emit(data, fmt="json")
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed == {"score": pytest.approx(0.5)}


def test_emit_yaml_mode_renders_numpy_scalars_without_raising(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = {"score": np.float32(0.5)}
    emit(data, fmt="yaml")
    out = capsys.readouterr().out
    parsed = yaml.safe_load(out)
    assert parsed == {"score": pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# emit() — stream resolved at call time
# ---------------------------------------------------------------------------


def test_emit_writes_to_stdout_replaced_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = io.StringIO()
    monkeypatch.setattr("sys.stdout", replacement)
    emit({"status": "ok"})
    assert replacement.getvalue() == "status=ok\n"


# ---------------------------------------------------------------------------
# exit_code_for()
# ---------------------------------------------------------------------------


def test_exit_code_for_file_not_found_error_is_not_found() -> None:
    assert exit_code_for(FileNotFoundError("missing.onnx")) == EXIT_NOT_FOUND


def test_exit_code_for_other_exception_is_error() -> None:
    assert exit_code_for(ValueError("bad input")) == EXIT_ERROR
