"""RAG-Enterprise-Intelligence source package root.

Every subpackage (ingestion, parsing, chunking, ...) is imported with an
absolute `src.<subpackage>` path from the repository root — e.g.
`from src.parsing.registry import get_parser_for` — so no PYTHONPATH
manipulation is needed beyond running Python from the repo root.
"""
