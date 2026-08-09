"""CLI entry point: python -m src.ingestion.cli --input documents/hr --domain hr

Runs the ingestion pipeline over a directory and prints a summary. Intended
for local runs and as the body of a future Airflow/scheduled job — kept free
of any framework-specific code so it can be wrapped either way.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.ingestion.pipeline import IngestionPipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a directory of documents into the RAG pipeline."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Directory of source documents to ingest."
    )
    parser.add_argument(
        "--domain", required=True, help="Knowledge domain / namespace, e.g. hr, legal, banking."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument(
        "--no-recursive", action="store_true", help="Only scan the top level of --input."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    pipeline = IngestionPipeline(processed_dir=args.processed_dir, metadata_dir=args.metadata_dir)
    results = pipeline.ingest_directory(args.input, args.domain, recursive=not args.no_recursive)

    counts = Counter(r.status.value for r in results)
    print(f"\nIngestion summary for domain='{args.domain}' ({args.input}):")
    for status, count in sorted(counts.items()):
        print(f"  {status:<18} {count}")
    print(f"  {'total':<18} {len(results)}")

    errors = [r for r in results if r.status.value == "error"]
    if errors:
        print("\nErrors:", file=sys.stderr)
        for r in errors:
            print(f"  {r.doc_id}: {r.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
