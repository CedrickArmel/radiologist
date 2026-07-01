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


def test_logged_artifacts_and_collection_member_importable_from_public_api():
    from radiologist.registry import CollectionMember, LoggedArtifacts

    assert LoggedArtifacts(
        det_qualified_name="entity/project/model-abc123:best",
        mcd_qualified_name="entity/project/model-abc123-mcd:best",
        run_id="abc123",
    )
    assert CollectionMember(
        qualified_name="entity/project/model-abc123:best", aliases=["best"]
    )


def test_promote_result_carries_alias_field():
    from radiologist.registry.models import PromoteResult

    result = PromoteResult(
        det_qualified_name="entity/project/model-abc123:staging",
        mcd_qualified_name="entity/project/model-abc123-mcd:staging",
        alias="staging",
    )
    assert result.alias == "staging"


def test_collection_lister_list_collection_artifacts_raises_not_implemented():
    from radiologist.registry.collection import _WandbCollectionLister

    with pytest.raises(NotImplementedError):
        _WandbCollectionLister().list_collection_artifacts("model", "det-collection")


def test_wandb_registry_log_model_artifacts_raises_not_implemented():
    from unittest.mock import MagicMock

    from radiologist.registry.models import ExportResult
    from radiologist.registry.wandb_registry import WandbRegistry

    export_result = ExportResult(
        det_path="/tmp/model.onnx",
        mcd_path="/tmp/model_mcd.onnx",
        run_id="abc123",
        input_shape=(1, 3, 224, 224),
        classes=["normal", "abnormal"],
    )
    with pytest.raises(NotImplementedError):
        WandbRegistry().log_model_artifacts(export_result, MagicMock(), "best.ckpt")


def test_wandb_registry_list_collection_artifacts_raises_not_implemented():
    from radiologist.registry.wandb_registry import WandbRegistry

    with pytest.raises(NotImplementedError):
        WandbRegistry().list_collection_artifacts("model", "det-collection")


def test_model_registry_protocol_declares_new_methods():
    from radiologist.registry.interface import ModelRegistry

    for name in (
        "log_model_artifacts",
        "list_collection_artifacts",
        "promote",
        "transition_to_production",
    ):
        assert hasattr(ModelRegistry, name)
