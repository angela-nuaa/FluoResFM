"""Audit the 12 canonical external BioTISR super-resolution protocol rows.

This script never changes the source workbook or data.  It converts the
author-machine paths in ``dataset_test-v2.xlsx`` to a user-supplied local
BioTISR transformed-data root, then writes a JSON readiness report.  Store the
report under an ignored experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath


CANONICAL_IDS = {
    f"biotisr-{structure}-sr-{level}"
    for structure in ("ccp", "factin", "lysosome", "mt")
    for level in (1, 2, 3)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap(author_path: str, data_root: Path) -> Path:
    parts = PureWindowsPath(author_path).parts
    try:
        start = next(i for i, part in enumerate(parts) if part.lower() == "transformed")
    except StopIteration as error:
        raise ValueError(f"Expected a transformed-data path, got: {author_path}") from error
    return data_root.joinpath(*parts[start + 1 :])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "repos" / "fluoresfm" / "dataset_test-v2.xlsx",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Local directory corresponding to the workbook's transformed/ directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit("pandas and openpyxl are required; use the environment.yml environment.") from error

    metadata = args.metadata.resolve()
    data_root = args.data_root.resolve()
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)

    frame = pd.read_excel(metadata)
    rows = frame[frame["id"].isin(CANONICAL_IDS)].copy()
    found = set(rows["id"])
    if found != CANONICAL_IDS:
        raise RuntimeError(f"Metadata IDs differ: missing={sorted(CANONICAL_IDS - found)}, extra={sorted(found - CANONICAL_IDS)}")

    report_rows = []
    for _, row in rows.sort_values("id").iterrows():
        index_path = remap(str(row["path_index"]), data_root)
        input_dir = remap(str(row["path_lr"]), data_root)
        reference_dir = remap(str(row["path_hr"]), data_root)
        filenames = []
        if index_path.is_file():
            filenames = [name for name in index_path.read_text(encoding="utf-8").splitlines() if name]
        missing_input = [name for name in filenames if not (input_dir / name).is_file()]
        missing_reference = [name for name in filenames if not (reference_dir / name).is_file()]
        report_rows.append(
            {
                "id": row["id"],
                "task": row["task#"],
                "structure": row["structure#"],
                "input_dir": str(input_dir),
                "reference_dir": str(reference_dir),
                "index": str(index_path),
                "index_exists": index_path.is_file(),
                "input_dir_exists": input_dir.is_dir(),
                "reference_dir_exists": reference_dir.is_dir(),
                "num_indexed_images": len(filenames),
                "missing_input": missing_input,
                "missing_reference": missing_reference,
                "ready": bool(filenames) and not missing_input and not missing_reference,
            }
        )

    report = {
        "metadata": {"path": str(metadata), "sha256": sha256_file(metadata)},
        "data_root": str(data_root),
        "canonical_ids": sorted(CANONICAL_IDS),
        "ready_count": sum(row["ready"] for row in report_rows),
        "rows": report_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: {report['ready_count']}/{len(report_rows)} datasets ready")


if __name__ == "__main__":
    main()
