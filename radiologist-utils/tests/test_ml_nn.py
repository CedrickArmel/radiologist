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

import pytest
import torch
import torch.nn as nn

from radiologist.utils.ml import initialize_weights


class _SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.linear = nn.Linear(8, 2)

    def forward(self, x):
        return x


def test_initialize_weights_normal_changes_conv_weights():
    torch.manual_seed(0)
    net = _SmallNet()
    original_conv = net.conv.weight.data.clone()
    torch.manual_seed(0)
    initialize_weights(net, dist="normal")
    assert not torch.equal(net.conv.weight.data, original_conv)


def test_initialize_weights_normal_changes_linear_weights():
    torch.manual_seed(0)
    net = _SmallNet()
    original_linear = net.linear.weight.data.clone()
    torch.manual_seed(0)
    initialize_weights(net, dist="normal")
    assert not torch.equal(net.linear.weight.data, original_linear)


def test_initialize_weights_uniform_changes_conv_weights():
    torch.manual_seed(0)
    net = _SmallNet()
    original_conv = net.conv.weight.data.clone()
    torch.manual_seed(0)
    initialize_weights(net, dist="uniform")
    assert not torch.equal(net.conv.weight.data, original_conv)


def test_initialize_weights_uniform_changes_linear_weights():
    torch.manual_seed(0)
    net = _SmallNet()
    original_linear = net.linear.weight.data.clone()
    torch.manual_seed(0)
    initialize_weights(net, dist="uniform")
    assert not torch.equal(net.linear.weight.data, original_linear)


def test_initialize_weights_normal_is_deterministic_given_seed():
    torch.manual_seed(7)
    net1 = _SmallNet()
    initialize_weights(net1, dist="normal")

    torch.manual_seed(7)
    net2 = _SmallNet()
    initialize_weights(net2, dist="normal")

    assert torch.equal(net1.conv.weight.data, net2.conv.weight.data)
    assert torch.equal(net1.linear.weight.data, net2.linear.weight.data)


def test_initialize_weights_uniform_is_deterministic_given_seed():
    torch.manual_seed(7)
    net1 = _SmallNet()
    initialize_weights(net1, dist="uniform")

    torch.manual_seed(7)
    net2 = _SmallNet()
    initialize_weights(net2, dist="uniform")

    assert torch.equal(net1.conv.weight.data, net2.conv.weight.data)
    assert torch.equal(net1.linear.weight.data, net2.linear.weight.data)


def test_initialize_weights_raises_for_unknown_dist():
    net = _SmallNet()
    with pytest.raises(ValueError, match="dist"):
        initialize_weights(net, dist="invalid")


def test_initialize_weights_conv_bias_zeroed():
    net = _SmallNet()
    initialize_weights(net, dist="normal")
    assert torch.all(net.conv.bias.data == 0)


def test_initialize_weights_linear_bias_zeroed():
    net = _SmallNet()
    initialize_weights(net, dist="normal")
    assert torch.all(net.linear.bias.data == 0)
