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

"""Sentinel handles for optional third-party dependencies.

Each name is set to the imported module when available, or None when the
extra is not installed.  Callers check the sentinel before use and raise
RuntimeError naming the missing extra.
"""

try:
    import fastapi as _fastapi  # type: ignore[import-untyped]
except ImportError:
    _fastapi = None  # type: ignore[assignment]

try:
    import uvicorn as _uvicorn  # type: ignore[import-untyped]
except ImportError:
    _uvicorn = None  # type: ignore[assignment]

try:
    import prometheus_client as _prometheus_client  # type: ignore[import-untyped]
except ImportError:
    _prometheus_client = None  # type: ignore[assignment]
