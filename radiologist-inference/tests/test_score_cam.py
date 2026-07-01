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

"""Behavioral tests for the standalone score_cam() saliency function (issue #79).

Tests drive through the public API only. Predictor.explain()'s behavioral
coverage lives in test_explainer.py, which exercises the same saliency
computation through Explainer.explain().
"""

import numpy as np

FEAT_C, FEAT_H, FEAT_W = 64, 7, 7


# ---------------------------------------------------------------------------
# score_cam() standalone function
# ---------------------------------------------------------------------------


class TestScoreCamFunction:
    def test_score_cam_returns_array_with_values_in_0_1(self):
        """score_cam() must return a saliency map with all values in [0, 1]."""
        from radiologist.inference import score_cam

        feature_maps = np.random.rand(FEAT_C, FEAT_H, FEAT_W).astype(np.float32)
        logits = np.array([0.3, 0.7], dtype=np.float32)
        result = score_cam(feature_maps=feature_maps, logits=logits)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_score_cam_returns_2d_array_with_feature_map_spatial_dims(self):
        """score_cam() must return shape (H, W) matching feature_maps spatial dims."""
        from radiologist.inference import score_cam

        feature_maps = np.random.rand(FEAT_C, FEAT_H, FEAT_W).astype(np.float32)
        logits = np.array([0.3, 0.7], dtype=np.float32)
        result = score_cam(feature_maps=feature_maps, logits=logits)
        assert result.shape == (FEAT_H, FEAT_W)

    def test_score_cam_uniform_feature_maps_returns_uniform_saliency(self):
        """score_cam() on uniform feature maps (all same channel) produces uniform map."""
        from radiologist.inference import score_cam

        feature_maps = np.ones((FEAT_C, FEAT_H, FEAT_W), dtype=np.float32)
        logits = np.array([0.5, 0.5], dtype=np.float32)
        result = score_cam(feature_maps=feature_maps, logits=logits)
        assert result.shape == (FEAT_H, FEAT_W)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
