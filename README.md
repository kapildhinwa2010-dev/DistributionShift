# DistributionShift

DistributionShift is a small Python library for comparing singular-value spectra of named two-dimensional matrices. It reports per-layer spectral differences between two sets of matrices and includes basic spectral summaries and bottleneck ranking.

## Status and scope

The repository contains a single Python module (version 0.1.0), packaging metadata, and a focused test suite. It operates on matrices supplied by the caller; it does not load a model, collect activations, determine whether a dataset has shifted, or evaluate downstream task performance. Treat its output as a comparison of matrix spectra, not as a general measure of model or data-distribution change.

## Features

- Compute singular values for a two-dimensional matrix.
- Summarize a matrix with its shape, Frobenius norm, spectral norm, condition number, and effective rank.
- Summarize named matrices and rank their layers by effective rank.
- Compare the spectral profiles of two matrices with `spectrum_distance`.
- Compare matrices under common layer names with `shift_report`, which also identifies the common layer with the largest distance.

For the distance, each matrix's singular values are divided by their maximum, transformed with `log1p`, sampled at evenly spaced quantiles (16 by default), and normalized before an L1 difference is calculated. Consequently, uniform scaling is intentionally ignored. `effective_rank` uses the default threshold of 0.99 of the sum of singular values (not squared singular-value energy).

## Requirements and installation

Python 3.9 or newer and NumPy 1.24 or newer are declared project requirements. From a checkout of this repository, install the project with:

```bash
python -m pip install .
```

To install the dependencies used for development and tests instead, run:

```bash
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np
from distributionshift import shift_report

baseline = {
    "encoder": np.eye(4),
    "projection": np.diag([4.0, 2.0, 1.0, 0.5]),
}
candidate = {
    "encoder": np.eye(4),
    "projection": np.diag([4.0, 1.0, 0.25, 0.1]),
}

report = shift_report(baseline, candidate)
print(report)  # Per-common-layer distances and the "most_shifted" layer name
```

The matrices in the two dictionaries are matched by key. Inputs are converted to NumPy `float64` arrays for spectral calculations.

## API

- `layer_spectrum(matrix)` returns singular values in descending order. The input must be two-dimensional.
- `effective_rank(matrix, energy=0.99)` returns the smallest count whose cumulative singular values reach the requested fraction. `energy` must be in `(0, 1]`.
- `layer_report(matrix)` returns `shape`, `frobenius_norm`, `spectral_norm`, `condition_number`, and `effective_rank` for a non-empty two-dimensional matrix. Numerically singular matrices are reported with an infinite condition number.
- `analyze_model(layers)` applies `layer_report` to each entry in a non-empty `{name: matrix}` dictionary.
- `find_bottlenecks(reports, top_k=1)` returns names ordered by ascending effective rank; ties are ordered by name.
- `spectrum_distance(baseline, candidate, bins=16)` returns a non-negative distance between normalized spectral profiles. `bins` must be at least 2.
- `shift_report(baseline_layers, candidate_layers, bins=16)` returns distances for common names plus `most_shifted`. It raises `ValueError` if either dictionary is empty or there are no common names.

## Tests

Run the test suite with:

```bash
python -m pytest -q
```

On the cloned repository commit, the suite completed with **17 passed**. The CI workflow is configured to run the tests on Python 3.10, 3.11, and 3.12; this local result does not establish the outcome of a separate hosted CI run.

## Limitations

- The comparison measures differences in the shape of singular-value profiles only; equal profiles under uniform rescaling have zero distance.
- A distance does not explain the cause of a difference or establish that accuracy, task behavior, or data quality changed.
- Only keys present in both layer dictionaries are compared. Added or removed names are not included in the report.
- Quantile sampling (16 points by default) compresses the spectrum and can omit fine-grained details.
- The module accepts matrices rather than model objects or activation streams; data extraction and interpretation are outside its scope.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for the complete terms and copyright notice.
