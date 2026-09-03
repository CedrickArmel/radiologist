---
name: issue-test-scope-incomplete
description: An issue's "Test updates (owned by this slice)" list can omit test call sites that break anyway because of a signature change — grep for the old signature across the whole test suite before trusting the list.
metadata:
  type: feedback
---

When an issue changes a public method's signature (e.g. `from_path(det_path, mcd_path=None)`
→ `from_path(model_path)`), its "Technical notes" section may enumerate only the test files/line
ranges the architect explicitly reviewed — not every call site that breaks as a mechanical
consequence of the signature change. In radiologist#141, `TestMCDropoutFromPathMeanStdValidation`
in `test_mc_dropout.py` called `MCDropoutPredictor.from_path(det_path=..., mcd_path=...)` directly
and was not mentioned anywhere in the issue body, yet it broke immediately once the signature
changed.

**Why:** issue authors write the test-update list from memory/diff-review at spec time; a
grep-verifiable mechanical break can slip through even in a careful spec. Trusting the list
literally leaves `pytest` red despite believing you did everything asked.

**How to apply:** after implementing a signature change (rename/drop kwarg, rename method), run
`grep -rn "<old_signature_pattern>"` across the whole test directory (not just the files the issue
names) before declaring green. Fix every hit, even ones the issue didn't call out — the AC
"pytest green" always wins over an incomplete enumeration. See also
[[feedback_shared_base_class_seam_in_decomposition_epics]] for the same principle applied to
production code seams.
