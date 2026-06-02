# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes
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

import random

import numpy as np
import torch

from radiologist.utils.ml import get_seeded_generator, seed_worker, set_seed


def test_set_seed_makes_torch_draws_reproducible():
    set_seed(42)
    t1 = torch.rand(4).tolist()
    set_seed(42)
    t2 = torch.rand(4).tolist()
    assert t1 == t2


def test_set_seed_makes_numpy_draws_reproducible():
    set_seed(42)
    a1 = np.random.rand(4).tolist()
    set_seed(42)
    a2 = np.random.rand(4).tolist()
    assert a1 == a2


def test_set_seed_makes_random_draws_reproducible():
    set_seed(42)
    r1 = [random.random() for _ in range(4)]
    set_seed(42)
    r2 = [random.random() for _ in range(4)]
    assert r1 == r2


def test_set_seed_none_draws_fresh_seed_each_call():
    set_seed(None)
    s1 = torch.initial_seed()
    set_seed(None)
    s2 = torch.initial_seed()
    assert s1 != s2


def test_get_seeded_generator_returns_torch_generator():
    gen = get_seeded_generator(7)
    assert isinstance(gen, torch.Generator)


def test_get_seeded_generator_produces_deterministic_first_draw():
    gen1 = get_seeded_generator(7)
    gen2 = get_seeded_generator(7)
    t1 = torch.rand(1, generator=gen1).item()
    t2 = torch.rand(1, generator=gen2).item()
    assert t1 == t2


def test_get_seeded_generator_different_seeds_produce_different_draws():
    gen_a = get_seeded_generator(7)
    gen_b = get_seeded_generator(99)
    t_a = torch.rand(1, generator=gen_a).item()
    t_b = torch.rand(1, generator=gen_b).item()
    assert t_a != t_b


def test_seed_worker_seeds_numpy_and_random_from_torch_initial_seed():
    # Simulate what DataLoader does in a worker: set torch seed, call seed_worker.
    torch.manual_seed(12345)
    seed_worker(0)
    np_draw_1 = np.random.rand(4).tolist()
    rand_draw_1 = [random.random() for _ in range(4)]

    torch.manual_seed(12345)
    seed_worker(0)
    np_draw_2 = np.random.rand(4).tolist()
    rand_draw_2 = [random.random() for _ in range(4)]

    assert np_draw_1 == np_draw_2
    assert rand_draw_1 == rand_draw_2
