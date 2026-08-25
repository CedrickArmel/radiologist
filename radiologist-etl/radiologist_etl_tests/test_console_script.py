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

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ETL_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = ETL_ROOT / "pyproject.toml"


def test_console_script_entry_point_targets_hydra_main():
    content = PYPROJECT_PATH.read_text()
    match = re.search(r'^radiologist-etl\s*=\s*"([^"]+)"', content, flags=re.MULTILINE)

    assert match is not None
    assert match.group(1) == "radiologist.etl.prefect_pipelines:main"


def test_console_script_help_composes_and_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "radiologist.etl.prefect_pipelines", "--help"],
        cwd=ETL_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Powered by Hydra" in result.stdout
