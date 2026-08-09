from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_asset_manifest_example_is_valid_json(self) -> None:
        payload = json.loads((ROOT / "assets" / "manifest.local.example.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "fluoresfm-assets/v1")
        self.assertEqual(len(payload["resources"]), 2)

    def test_repository_contracts(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_repository.py", "--workspace-root", "."],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
