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

import os
import random

import numpy as np
import torch


def set_seed(
    seed: int | None = None,
    cudnn_backend: bool = False,
    use_deterministic_algorithms: bool = False,
    warn_only: bool = True,
) -> None:
    """Seed python/numpy/torch for reproducibility.

    Args:
        seed: Integer seed. None draws a fresh torch seed.
        cudnn_backend: Whether to set deterministic cuDNN backend.
        use_deterministic_algorithms: Whether to enforce deterministic algorithms.
        warn_only: If True, warn instead of error on non-deterministic ops.
    """

    seed = (torch.default_generator.seed() if seed is None else seed) % (2**32)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = cudnn_backend
    torch.backends.cudnn.benchmark = not cudnn_backend

    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def seed_worker(worker_id: int) -> None:
    """Seed numpy and random in a DataLoader worker from torch.initial_seed().

    Args:
        worker_id: The worker index (provided automatically by DataLoader).
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_seeded_generator(seed: int) -> torch.Generator:
    """Return a torch.Generator manually seeded with seed mod 2**32.

    Args:
        seed: Integer seed value.

    Returns:
        A seeded torch.Generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed % (2**32))
    return generator
