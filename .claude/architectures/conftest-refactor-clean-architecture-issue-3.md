## ✨ Slice B — `test_datamodule.py` simplification (use the `dm` fixture)

### Context

Every common-case test in `test_datamodule.py` lists 8+ fixture parameters and threads them through the local `_make_dm` factory. The seam (#1) added a composed function-scoped `dm` fixture that wires the default configuration. This slice rewrites the common-case tests to take the single `dm` parameter and deletes the now-redundant `_make_dm` factory. Tests that intentionally use non-default constructor args keep building their own datamodule from the individual fixtures. No assertion changes. Requires: #1. See the epic spec for context.

### User story

As a **maintainer of the core datamodule tests**, I want **the common-case datamodule built by one fixture** so that **a test reads as its scenario and assertion, not 8 lines of fixture wiring**.

### Acceptance criteria

<!-- Behavioral assertions are unchanged. Criteria assert the end state and continued green. -->

- [ ] Common-case datamodule tests (num_classes, setup-fit success, auto/explicit priors, dataloader keys/shapes/target indices, split sizes, splitter kwargs) build the datamodule through a single fixture parameter and still assert exactly what they did before.
- [ ] The local `_make_dm` factory function is removed from `test_datamodule.py`.
- [ ] Tests that exercise non-default configuration keep their own construction and still pass: the `shared_map` two-labels-one-class case, the `classes=None` derivation case, the `batch_size=1` epoch-length case, and the empty-root `FileNotFoundError` case (which needs `tmp_path`, not `shard_root`).
- [ ] The rank-0/rank-1 broadcast test still patches `records_reader` and asserts the same broadcast behavior, obtaining its datamodule(s) through the fixture or individual fixtures as appropriate.
- [ ] mypy clean; pytest green

### Out of scope

- Renaming test functions/classes or changing assertions.
- The merged `transform`/session-scope changes themselves (delivered in #1); this slice only consumes them.

### Technical notes

- `radiologist-core/tests/test_datamodule.py` — the `dm` fixture covers only the default config. Four tests must NOT use it: `test_num_classes_derived_from_label_map_when_classes_is_none` (omits `classes`), `test_setup_fit_raises_file_not_found_when_train_shards_missing` (uses an empty `tmp_path` root, not `shard_root`), `test_two_etl_labels_mapping_to_same_class_produce_same_index` (custom `shared_map`), and `test_train_epoch_length_equals_train_size_divided_by_batch_size` (`batch_size=1` with a custom loader partial). These continue to construct `WebDatasetDataModule` directly from the individual fixtures.
- The seam merged the transforms into one `transform` fixture; direct-construction tests should pass `transform` as both `train_transform` and `eval_transform`.
