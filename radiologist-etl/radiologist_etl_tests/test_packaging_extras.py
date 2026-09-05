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

"""Executable contract pinning radiologist-etl's optional-dependency taxonomy.

See issue #213: this module freezes the extras invariants that already hold
today. It parses ``radiologist-etl/pyproject.toml`` directly -- no
``radiologist.etl`` import -- so it stays green on a checkout where no extra
at all is installed.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - this repo pins Python 3.10
    import tomli as tomllib  # type: ignore[import-untyped,no-redef]

_SINGLE_DISTRIBUTION_EXTRAS = ("gcs", "prefect", "dask", "ray", "beam")
_PRODUCTION_READY_EXTRAS = ("gcs", "prefect", "dask", "beam")
_DISALLOWED_DEFAULT_DISTRIBUTIONS = (
    "prefect",
    "prefect-dask",
    "prefect-ray",
    "apache-beam",
)

_REQUIREMENT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+")


def _etl_pyproject_path() -> Path:
    return Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_etl_pyproject() -> Dict[str, Any]:
    with _etl_pyproject_path().open("rb") as handle:
        return tomllib.load(handle)


def _normalise(distribution_name: str) -> str:
    return distribution_name.lower().replace("_", "-")


def _requirement_distribution_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME_RE.match(requirement.strip())
    if not match:
        raise ValueError(f"Could not parse a distribution name from {requirement!r}")
    return _normalise(match.group(0))


def _extra_distribution_names(data: Dict[str, Any], extra: str) -> Set[str]:
    requirements: List[str] = data["project"]["optional-dependencies"][extra]
    return {_requirement_distribution_name(req) for req in requirements}


@pytest.mark.parametrize("extra", _SINGLE_DISTRIBUTION_EXTRAS)
def test_execution_backend_extra_names_exactly_one_distribution(
    extra: str,
) -> None:
    data = _load_etl_pyproject()
    distributions = _extra_distribution_names(data, extra)
    assert len(distributions) == 1, (
        f"extra {extra!r} must name exactly one distribution, " f"got {distributions!r}"
    )


def test_default_dependencies_exclude_orchestrator_and_backends() -> None:
    data = _load_etl_pyproject()
    default_names = {
        _requirement_distribution_name(req) for req in data["project"]["dependencies"]
    }
    disallowed = {_normalise(name) for name in _DISALLOWED_DEFAULT_DISTRIBUTIONS}
    offending = default_names & disallowed
    assert not offending, (
        "radiologist-etl's default dependencies must not name an "
        f"orchestrator or execution-backend distribution, found {offending!r}"
    )


@pytest.mark.parametrize("extra", _PRODUCTION_READY_EXTRAS)
def test_all_extra_is_superset_of_each_production_ready_extra(
    extra: str,
) -> None:
    data = _load_etl_pyproject()
    all_names = _extra_distribution_names(data, "all")
    extra_names = _extra_distribution_names(data, extra)
    missing = extra_names - all_names
    assert not missing, (
        "the aggregate 'all' extra is missing distributions required by "
        f"extra {extra!r}: {missing!r}"
    )
