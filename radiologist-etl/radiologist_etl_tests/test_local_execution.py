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

"""Behavioral tests for the local execution primitives (#183: default_workers,
chunked, local_mapper) — the in-process-pool BatchMapper default."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest

# Module-level worker functions below cross a real process boundary (spawn
# context) when local_mapper dispatches them. The spawned child starts a
# fresh interpreter and must be able to import this test module by its
# dotted path; only the parent process benefits from pytest's sys.path
# manipulation, so PYTHONPATH must carry it explicitly for the child.
_TESTS_PARENT = str(Path(__file__).resolve().parents[1])
if _TESTS_PARENT not in sys.path:
    sys.path.insert(0, _TESTS_PARENT)
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
_pythonpath_parts = [_TESTS_PARENT] + (
    _existing_pythonpath.split(os.pathsep) if _existing_pythonpath else []
)
os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(_pythonpath_parts))


def test_default_workers_matches_cpu_count():
    from radiologist.etl.execution import default_workers

    assert default_workers() == (os.cpu_count() or 1)


def test_chunked_splits_into_consecutive_groups_of_at_most_size():
    from radiologist.etl.execution import chunked

    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunked_last_chunk_may_be_shorter():
    from radiologist.etl.execution import chunked

    result = chunked(["a", "b", "c"], 5)

    assert result == [["a", "b", "c"]]


def test_chunked_of_empty_input_yields_empty_list():
    from radiologist.etl.execution import chunked

    assert chunked([], 3) == []


def test_chunked_rejects_non_positive_size():
    from radiologist.etl.execution import chunked

    with pytest.raises(ValueError):
        chunked([1, 2, 3], 0)


def _double(x: int) -> int:
    return x * 2


def test_local_mapper_returns_results_in_input_order():
    from radiologist.etl.execution import local_mapper

    mapper = local_mapper(_double, workers=2)

    assert mapper([5, 1, 4, 2, 3]) == [10, 2, 8, 4, 6]


def test_local_mapper_defaults_workers_to_default_workers():
    from radiologist.etl.execution import default_workers, local_mapper

    mapper = local_mapper(_double)

    assert mapper([1, 2, 3]) == [2, 4, 6]
    # default_workers() is used as the pool size; no direct introspection
    # point exists other than behavior, so just confirm it does not error
    # and default_workers() itself resolves to a positive int.
    assert default_workers() >= 1


def _sleep_and_track(item, counter, lock, max_seen):
    with lock:
        counter.value += 1
        if counter.value > max_seen.value:
            max_seen.value = counter.value
    time.sleep(0.15)
    with lock:
        counter.value -= 1
    return item


def test_local_mapper_never_exceeds_max_pending_outstanding_units():
    from radiologist.etl.execution import local_mapper

    manager = mp.Manager()
    counter = manager.Value("i", 0)
    max_seen = manager.Value("i", 0)
    lock = manager.Lock()

    import functools

    fn = functools.partial(
        _sleep_and_track, counter=counter, lock=lock, max_seen=max_seen
    )
    mapper = local_mapper(fn, workers=4, max_pending=2)

    results = mapper([1, 2, 3, 4, 5, 6])

    assert sorted(results) == [1, 2, 3, 4, 5, 6]
    assert max_seen.value <= 2
