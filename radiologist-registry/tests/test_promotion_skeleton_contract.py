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


def test_model_registry_protocol_declares_new_methods():
    from radiologist.registry.interface import ModelRegistry

    for name in (
        "log_model_artifacts",
        "list_collection_artifacts",
        "promote",
        "transition_to_production",
    ):
        assert hasattr(ModelRegistry, name)
