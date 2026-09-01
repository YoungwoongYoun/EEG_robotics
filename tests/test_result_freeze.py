import csv
import tempfile
import unittest
from pathlib import Path

from eeg_pipeline.experiments.result_freeze import read_csv, sha256_file


class ResultFreezeTests(unittest.TestCase):
    def test_sha256_file_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.bin"
            path.write_bytes(b"eeg-results")
            self.assertEqual(
                sha256_file(path),
                "ddb80d310e9033da327ca337763d0f6eee987a42c8f83b07145f4ca00f6a6eba",
            )

    def test_read_csv_requires_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                read_csv(Path(temporary) / "missing.csv")

    def test_read_csv_preserves_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("seed", "accuracy"))
                writer.writeheader()
                writer.writerow({"seed": 0, "accuracy": 0.5})
            self.assertEqual(read_csv(path), [{"seed": "0", "accuracy": "0.5"}])


if __name__ == "__main__":
    unittest.main()
