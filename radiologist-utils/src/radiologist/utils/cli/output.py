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

"""Machine-readable output helpers shared by every ``radiologist`` CLI command.

``emit()`` is the single primitive every CLI command calls to produce its
final stdout record — see the module docstring in ``radiologist.utils.cli``
for the full behavioural contract (kv/json/yaml rendering rules, numpy
scalar normalisation, stream resolution at call time).
"""

import json
import os
import sys
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence, TextIO, Tuple

import numpy as np

try:
    import yaml as _yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised via sentinel patching
    _yaml = None  # type: ignore[assignment]

OUTPUT_FORMATS: Tuple[str, ...] = ("kv", "json", "yaml")
DEFAULT_OUTPUT_FORMAT: str = "kv"
OUTPUT_ENV_VAR: str = "RADIOLOGIST_OUTPUT"

_YAML_MISSING_MSG = (
    "PyYAML is required for yaml output. "
    "Install with: pip install 'radiologist-utils[cli]'"
)

__all__ = [
    "OUTPUT_FORMATS",
    "DEFAULT_OUTPUT_FORMAT",
    "OUTPUT_ENV_VAR",
    "resolve_format",
    "emit",
]


def resolve_format(fmt: Optional[str] = None) -> str:
    """Resolve the effective output format.

    Resolution order: explicit ``fmt`` argument, then the
    ``RADIOLOGIST_OUTPUT`` environment variable, then
    :data:`DEFAULT_OUTPUT_FORMAT`. Both the argument and the environment
    variable are normalised (stripped and lower-cased) before validation.

    Args:
        fmt: Format requested via a CLI flag, taking precedence over the
            ``RADIOLOGIST_OUTPUT`` environment variable and the default.

    Returns:
        One of :data:`OUTPUT_FORMATS`.

    Raises:
        ValueError: If the resolved value is not one of
            :data:`OUTPUT_FORMATS`.
    """
    candidate = fmt if fmt is not None else os.environ.get(OUTPUT_ENV_VAR)
    if candidate is None:
        return DEFAULT_OUTPUT_FORMAT
    normalized = candidate.strip().lower()
    if normalized not in OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output format: {candidate!r}. "
            f"Must be one of {OUTPUT_FORMATS}."
        )
    return normalized


def _scalarize(value: Any) -> Any:
    """Normalise a single leaf value, converting numpy scalars to Python."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _scalarize_deep(value: Any) -> Any:
    """Recursively normalise numpy scalars while preserving structure."""
    value = _scalarize(value)
    if isinstance(value, Mapping):
        return {key: _scalarize_deep(sub) for key, sub in value.items()}
    if _is_sequence(value):
        return [_scalarize_deep(item) for item in value]
    return value


def _flatten_into(key: str, value: Any, result: "OrderedDict[str, Any]") -> None:
    value = _scalarize(value)
    if isinstance(value, Mapping):
        if not value:
            result[key] = None
            return
        for sub_key, sub_value in value.items():
            _flatten_into(f"{key}.{sub_key}", sub_value, result)
    elif _is_sequence(value):
        if not value:
            result[key] = None
            return
        for index, item in enumerate(value):
            _flatten_into(f"{key}[{index}]", item, result)
    else:
        result[key] = value


def _flatten(data: Mapping[str, Any], prefix: str = "") -> "OrderedDict[str, Any]":
    """Flatten ``data`` into leaves only, keyed by dotted/indexed paths.

    Args:
        data: Mapping to flatten.
        prefix: Key prefix prepended (dot-joined) to every top-level key.

    Returns:
        An ordered mapping of flattened key -> leaf value, preserving
        insertion order.
    """
    result: "OrderedDict[str, Any]" = OrderedDict()
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        _flatten_into(full_key, value, result)
    return result


def _render_kv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def emit(
    data: Mapping[str, Any],
    fmt: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> None:
    """Write ``data`` to ``stream`` in the resolved output format.

    Args:
        data: Mapping of result fields to serialize.
        fmt: Explicit output format, resolved via :func:`resolve_format` when
            not given.
        stream: Destination stream, defaults to ``sys.stdout`` resolved at
            call time (not bound at import time, so it plays nicely with
            ``capsys``/``CliRunner`` which swap ``sys.stdout`` after import).
    """
    resolved = resolve_format(fmt)
    stream = stream or sys.stdout
    if resolved == "kv":
        for key, value in _flatten(data).items():
            stream.write(f"{key}={_render_kv_value(value)}\n")
    elif resolved == "json":
        payload = _scalarize_deep(data)
        stream.write(json.dumps(payload) + "\n")
    else:  # yaml
        if _yaml is None:
            raise RuntimeError(_YAML_MISSING_MSG)
        payload = _scalarize_deep(data)
        stream.write(_yaml.safe_dump(payload, explicit_start=True))
