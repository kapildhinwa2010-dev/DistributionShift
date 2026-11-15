# DistributionShift

[![CI](https://github.com/kapildhinwa2010-dev/DistributionShift/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DistributionShift/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/DistributionShift/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DistributionShift/actions/workflows/codeql.yml)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Detect distribution shift between two layer sets by comparing singular-value spectra.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

For each layer name present in two sets of matrices (e.g. weights or
activation covariances before and after a data change),
spectrum_distance() compares the shapes of their singular-value
spectra: log-transformed, scale-normalized, quantile-binned, then
sum-of-absolute-differences. shift_report() runs this per layer and
names the most-shifted layer. The base spectral API (effective rank,
condition number, bottleneck ranking) is shared with BottleneckTest.

## Install

    pip install -r requirements.txt

## Quick start

    from distributionshift import shift_report

    report = shift_report(baseline_layers, candidate_layers)
    print(report["most_shifted"], report[report["most_shifted"]])

## API

- layer_spectrum, effective_rank, layer_report, analyze_model,
  find_bottlenecks (shared spectral base)
- spectrum_distance(baseline, candidate, bins=16) - float >= 0
- shift_report(baseline_layers, candidate_layers, bins=16) - per-layer
  distances plus most_shifted

## Honest results and limitations

- The distance compares only the SHAPE of the spectral decay; overall
  scale differences are deliberately ignored. Two matrices that differ
  by a constant factor score 0.0.
- A high distance says the spectra differ, not why, and not whether
  task accuracy changed. No accuracy evaluation is included.
- Layers are matched by name; renamed or new layers are silently
  skipped (only common names are compared).
- Quantile binning (default 16) smooths fine spectral detail.

## License

MIT - see [LICENSE](LICENSE).
