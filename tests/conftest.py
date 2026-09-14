"""Shared pytest fixtures and path setup for the benchmark test suite."""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")

# The pipeline modules are plain scripts under src/, not an installed package.
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_data_dir(repo_root):
    """The small .mat datasets committed to the repo for local sanity checks."""
    return os.path.join(repo_root, "sample_data_medium")


@pytest.fixture(scope="session")
def config():
    """The default benchmark configuration module."""
    from configs import config as config_module

    return config_module
