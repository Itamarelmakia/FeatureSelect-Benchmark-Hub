# Dataset Hardness Analysis

This directory is a navigation pointer to the published Dataset Hardness / Article 1 material.

The Dataset Hardness study evaluates the difficulty of the 102 real-world benchmark datasets
used in this hub, providing a difficulty score per dataset that contextualizes the feature
selection benchmark results.

## Where to find the material

The Dataset Hardness pipeline is part of the main hub benchmark:

- **Core pipeline:** [`src/Main_FS.py`](../../src/Main_FS.py) and [`src/utilities.py`](../../src/utilities.py)
- **Config:** [`src/configs/config.py`](../../src/configs/config.py)
- **Data:** [`data/`](../../data/) (102 `.mat` datasets across multiple repositories)

The benchmark includes a High-Dimensional (HD) experiment branch (`process_fs_algorithm` with
`'FS'` mode in `Main_FS.py`) that computes dataset hardness alongside feature selection results
for ETree and AdaBoost.

## Article reference

To be updated when Article 1 is published.
