"""Shared pytest fixtures for the RAG-Enterprise-Intelligence test suite.

Component-specific fixtures (sample documents, mock embeddings, a stood-up
vector store, etc.) are added alongside each component's tests starting in
Milestone 3; this file holds only the fixtures every test subpackage needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root, resolved from this file's location."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """Path to the Hydra config directory, for tests that load real config groups."""
    return project_root / "configs"


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """A throwaway directory standing in for data/ in tests that write files,
    so tests never touch the real data/ tree.
    """
    d = tmp_path / "data"
    d.mkdir()
    return d
