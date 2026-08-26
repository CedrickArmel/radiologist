---
name: pypi-publishing-constraints
description: Hard constraints discovered when designing PyPI publishing for the radiologist monorepo — uv_build license globs, metapackage src dir, PyPI trusted-publishing rejecting reusable workflows, and GITHUB_TOKEN refs not firing workflows
metadata:
  type: project
---

Four verified blockers govern any PyPI-publishing design in this repo.

**1. `license = { file = "../LICENSE" }` makes every sub-package unbuildable.**
`uv build --package radiologist-<pkg>` fails with `Unsupported glob expression in: project.license-files — The parent directory operator ('..') at position 0 is not allowed`. Fix that is verified to work: `license = "MIT"` + `license-files = ["LICENSE"]` + a real LICENSE file copied inside each package dir. A symlink to `../LICENSE` is not a reliable substitute.

**2. The root `radiologist` metapackage is not structurally a dispatcher.**
`uv build --package radiologist` fails with `IO error ... src/radiologist: No such file or directory`. uv_build defaults `module-name` to the project name and requires the dir to exist. Verified fix: create `src/radiologist/.gitkeep`. The resulting wheel contains only the empty namespace dir — harmless.

**3. PyPI Trusted Publishing does NOT work from inside a reusable (`workflow_call`) workflow.**
PyPI validates the OIDC `job_workflow_ref` claim, which points at the *called* workflow, never the caller — so a publisher registered against the caller filename yields `invalid-publisher`. Tracked in pypi/warehouse#11096. Composite actions are likewise unsupported for the publish step.

**4. Refs (tags/branches) created with the default `GITHUB_TOKEN` do not trigger other workflows.**
Generalises beyond tags: any ref or event produced by the default token is inert as a trigger, by design, to prevent recursive workflow runs. So a "bump automation creates tag → separate tag-triggered publish.yml" chain silently never fires. The usual workaround is a fine-grained PAT purely to make the trigger fire — reject that. Preferred fix, adopted here: collapse the chain into ONE workflow run that reacts to the bump PR merging (`pull_request: types:[closed]`, filtered on `merged == true` + a `release/<pkg>-v<X.Y.Z>` head-branch convention) and then tags, builds, tests and publishes in that same run. Carry the package/version on the *branch name*, not the commit message — a merge subject can be edited at merge time, a head ref cannot. Tag last (after a successful upload) so a failed run leaves nothing to clean up before a retry.

**Why:** points 3 and 4 together are the single biggest constraint on any DRY/reusable-workflow packaging strategy — it caps how much of the release pipeline can be abstracted.

**How to apply:** everything (setup, test, build, verify) may live in reusable workflows or composite actions, but the `pypa/gh-action-pypi-publish` step must be inlined in a top-level workflow. The DRY-preserving workaround: one single top-level `publish` workflow shared by all distributions, with a dynamic `environment: ${{ needs.resolve.outputs.environment }}`; each PyPI project then registers the same repo+workflow filename and is disambiguated purely by environment name.

Related: [[pytest-layout]].
