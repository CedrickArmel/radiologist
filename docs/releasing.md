# Releasing — trusted publishers, GitHub Environments, and the seeded first release

`radiologist` is a `uv`-managed mono-repo of six PyPI distributions: the root meta-package
`radiologist`, plus five workspace members `radiologist-utils`, `radiologist-etl`,
`radiologist-core`, `radiologist-inference`, `radiologist-registry`. `.github/workflows/release.yml`
and `.github/workflows/publish.yml` automate the bump → PR → test → build → publish → tag chain, but
they publish over OIDC **trusted publishing** with no stored PyPI credentials. Trusted publishing
works by PyPI trusting a specific `(owner, repository, workflow file, environment)` tuple to mint an
OIDC token for a specific project — **nothing in CI can create that trust**. A human with PyPI account
access and repository admin access must configure it once per distribution, and this page is the
runbook for doing that, plus for seeding the very first `0.1.0` release of all six distributions.

This is manual, one-time (per distribution) setup. It is not something a maintainer repeats for every
release — only when adding a new distribution to the mono-repo, or bootstrapping the whole repo as
described below.

## Do this only after the release pipeline is on `main`

`.github/workflows/ci.yml`, `.github/workflows/release.yml` and `.github/workflows/publish.yml` must
already be merged before any of the steps below, because PyPI's pending-publisher registration records
the workflow **filename** (`publish.yml`). If that file is ever renamed, all six pending/trusted
publishers below must be edited by hand to match.

## Part A — PyPI pending trusted publishers (six times)

Because none of the six distribution names has ever been published, a *regular* trusted publisher
cannot be attached to any of them — PyPI only offers that once a project exists. Instead, register a
**pending publisher** for each one: pypi.org → **Your projects** → **Publishing** → **Add a new pending
publisher** (GitHub tab), and enter:

| Field             | Value              |
| ----------------- | ------------------ |
| PyPI Project Name | *(per row below)*  |
| Owner              | `CedrickArmel`     |
| Repository name    | `radiologist`      |
| Workflow name       | `publish.yml`      |
| Environment name    | *(per row below)*  |

| PyPI Project Name       | Environment name             |
| ----------------------- | ----------------------------- |
| `radiologist`           | `pypi-radiologist`             |
| `radiologist-utils`     | `pypi-radiologist-utils`       |
| `radiologist-etl`       | `pypi-radiologist-etl`         |
| `radiologist-core`      | `pypi-radiologist-core`        |
| `radiologist-inference` | `pypi-radiologist-inference`   |
| `radiologist-registry`  | `pypi-radiologist-registry`    |

**This table is the security boundary.** Every row shares the same owner, repository and workflow
file — only the environment name differs. A wrong environment name here lets one distribution's
publish job mint a token valid for another distribution's project. Before saving each row, cross-check
it against:

- the tag/environment mapping frozen in `scripts/release_bump.py` (`environment_name`, `release_tag`,
  `parse_release_branch_name`), and
- each package's own `pyproject.toml` `[tool.commitizen]` `tag_format` (`$version` for the root
  `radiologist`, `${version}-radiologist-<name>` for each of the five members).

Adding a new distribution to the mono-repo in the future means repeating this Part for that one new
distribution — the mapping is `pypi-<package>`, always.

## Part B — GitHub Environments (six times)

Repository → **Settings** → **Environments** → **New environment**, named **exactly** as the
right-hand column in the table above (`pypi-<package>`). For each of the six:

- Add **Required reviewers** = the maintainer. This is the last human checkpoint before an upload, and
  it costs one click per release.
- **Leave the deployment-branch rule unrestricted.** `publish.yml`'s `publish` job runs against the
  merge commit under a `pull_request: closed` event (or a `workflow_dispatch` ref); over-tightening
  this rule is a common way to make every release fail with an opaque "environment not allowed" error.
- **Add no secrets.** These environments exist purely to scope the `publish` job's `id-token: write`
  permission to one distribution at a time — the same pattern already used by the `github-pages`
  environment in `docs.yml`.

## Part C — Repository secrets (exactly one)

| Secret          | Purpose                                    | How to obtain                                                          |
| --------------- | ------------------------------------------- | ----------------------------------------------------------------------- |
| `CODECOV_TOKEN` | Coverage upload from `ci.yml`               | codecov.io → add the `radiologist` repository → copy the upload token   |

**There is no publishing secret, and there must never be one.** PyPI uploads authenticate over OIDC
(Part A + Part B); bump commits are signed by GitHub itself through its GraphQL commit API
(`release.yml`); the release tag is created with the default `GITHUB_TOKEN` (`publish.yml`'s `tag`
job). `CODECOV_TOKEN` is the only long-lived secret in the repository, and it plays no part in the
publish path.

## Part D — Seed the baseline release

Do this only after Parts A–C are complete for all six distributions.

Every distribution's manifest still reads `0.1.0` — nothing has ever been bumped, and the repository
has zero git tags, so commitizen has no baseline to diff against for any of the six. The first release
therefore does not go through a bump pull request at all: use `publish.yml`'s `workflow_dispatch`
escape hatch directly, which runs the same resolve → test → build → publish → tag chain and creates the
baseline tag as a side effect of the first successful upload.

For each distribution, in the dependency order below: **Actions** → **publish** → **Run workflow** on
`main`, with

- **package** = the distribution name
- **version** = `0.1.0`

Approve the environment gate when the run pauses on it, and **wait for the run to go green before
starting the next one**:

1. `radiologist-utils` — no `radiologist-*` dependencies
2. `radiologist-registry` — no `radiologist-*` dependencies
3. `radiologist-etl` — depends on `radiologist-utils`
4. `radiologist-core` — depends on `radiologist-etl`, `radiologist-registry`, `radiologist-utils`
5. `radiologist-inference` — depends on `radiologist-registry`
6. `radiologist` — the meta-package; its extras reference all five members

The order is dependency order, not a correctness requirement — the version constraints between
distributions are plain PEP 440 and are not checked at upload time. Publishing out of order simply
leaves an installable-but-broken distribution on PyPI until its dependency lands, and **PyPI versions
can never be re-uploaded**, so getting the order right the first time avoids burning a version number.

Each successful upload converts that distribution's *pending* publisher into a normal trusted
publisher automatically — no further PyPI-side configuration is needed after Part D.

After all six runs, confirm the six tags exist on `main`:

```text
0.1.0
0.1.0-radiologist-utils
0.1.0-radiologist-etl
0.1.0-radiologist-core
0.1.0-radiologist-inference
0.1.0-radiologist-registry
```

Every subsequent `cz bump` (via `release.yml`) diffs against one of these.

## Part E — Verify from outside the repository

In a throwaway virtualenv on a machine with no checkout of this repository:

```bash
pip install radiologist                 # expect: no radiologist-* component pulled in
pip install "radiologist[inference]"    # expect: radiologist-inference + fastapi/uvicorn/typer/wandb
pip install "radiologist[registry]"     # expect: radiologist-registry + wandb + typer, and no torch
python -c "import radiologist.inference, radiologist.registry"
```

The last line is the real proof that the PEP 420 implicit namespace survived packaging: no
distribution ships a `radiologist/__init__.py`, so two independently installed distributions must both
be importable under `radiologist.`. The root meta-package ships one inert `radiologist/.gitkeep`
marker — confirm it is present in the installed wheel and harmless, not an `__init__.py`.

Then confirm the ongoing path end to end: trigger a bump for one distribution via `release.yml`, review
the resulting pull request (the bump commit must show **Verified**), rebase-merge it, and watch the
single `publish.yml` run test, build, upload and tag without any further manual action beyond the
environment approval click.

## Recovery

- **A `0.1.0` (or any) publish run fails before the upload step.** The release tag is only created
  after a successful upload (`publish.yml`'s `tag` job depends on `publish`), so a failed run before
  that point leaves nothing to clean up. Fix the cause and re-run the same manual dispatch or re-merge
  is not needed — just re-dispatch.
- **A publish run fails *after* PyPI accepted the upload.** PyPI versions are immutable — do not retry
  the same version. Bump to the next patch version through the normal `release.yml` path instead.
  Re-running the dispatch for the already-uploaded version is still safe for the `tag` job alone, which
  is a no-op when the ref already exists.
- **PyPI/version badges render "not found".** Expected and self-resolving until Part D completes for
  that distribution.

## Adding a seventh distribution later

Repeat Part A and Part B for the new distribution only (`pypi-<new-package>` environment name,
same owner/repository/workflow), then run Part D's `workflow_dispatch` once for it at whatever version
its manifest currently holds. Parts C and E do not change.

## Out of scope

- Organisation-level PyPI publishing settings or two-factor authentication policy.
- TestPyPI — Part D is the rehearsal, run directly against PyPI, because pending publishers are
  themselves the safety mechanism for a first upload.
- Automating any part of Parts A–C. PyPI's pending-publisher flow and GitHub's environment settings are
  deliberately human-gated; automating a security boundary that is configured once per distribution
  would cost more than it saves.
