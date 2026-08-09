"""Base parser interface. Every format-specific parser in src/parsing/ implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.parsing.schemas import ParsedDocument


class BaseParser(ABC):
    """Common interface for all format parsers.

    Subclasses only need to implement :meth:`parse`. ``supported_extensions``
    drives auto-selection in ``src/parsing/registry.py``.
    """

    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: str | Path) -> ParsedDocument:
        """Parse a source file into a ParsedDocument. Must not raise on
        malformed-but-present files — collect issues in ``parser_warnings``
        and return the best-effort result instead, so one bad file doesn't
        halt a batch ingestion run.
        """
        raise NotImplementedError

    def _read_path(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Source file not found: {p}")
        if not p.is_file():
            raise IsADirectoryError(f"Expected a file, got a directory: {p}")
        return p
