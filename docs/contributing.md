# Contributing — Documentation Conventions

This project documents its public API using
[Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
enforced via `flake8-docstrings` with `docstring-convention = google` in
`.flake8`.

## Example

```python
def normalize(image: "np.ndarray", mean: float, std: float) -> "np.ndarray":
    """Normalize a single-channel image array.

    Args:
        image: Input image array, shape (H, W).
        mean: Mean pixel value used for centering.
        std: Standard deviation used for scaling.

    Returns:
        The normalized image array with the same shape as ``image``.

    Raises:
        ValueError: If ``std`` is zero.
    """
```

## Scope of the lint gate

Only **public** symbols (those exported via a package's `__init__.py` `__all__`,
plus public functions/classes/methods not prefixed with `_`) are required to carry
a docstring. Private, `_`-prefixed symbols are exempt from the `D1xx` (missing
docstring) checks.
