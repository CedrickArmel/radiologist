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

"""Internal W&B seam for artifact alias management, used by WandbRegistry."""

from typing import List

from radiologist.registry.optional import _guard_wandb, _wandb


class _WandbAliasManager:
    """W&B seam for artifact alias management operations."""

    def get_aliases(self, artifact_path: str) -> List[str]:
        """Return a snapshot of the artifact's current alias list."""
        _guard_wandb()
        api = _wandb.Api()  # type: ignore[union-attr]
        art = api.artifact(artifact_path)
        return list(art.aliases)

    def set_alias(self, artifact_path: str, alias: str) -> None:
        """Add alias to the artifact; no-op if already present."""
        _guard_wandb()
        api = _wandb.Api()  # type: ignore[union-attr]
        art = api.artifact(artifact_path)
        if alias not in art.aliases:
            art.aliases.append(alias)
            art.save()

    def remove_alias(self, artifact_path: str, alias: str) -> None:
        """Remove alias from the artifact; no-op if absent."""
        _guard_wandb()
        api = _wandb.Api()  # type: ignore[union-attr]
        art = api.artifact(artifact_path)
        if alias in art.aliases:
            art.aliases.remove(alias)
            art.save()
