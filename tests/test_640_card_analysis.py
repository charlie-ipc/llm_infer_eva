import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.analysis import scan_640_card_intra_node as analysis


class Scan640CardAnalysisTests(unittest.TestCase):
    def test_batch_mapping_and_communication_payloads(self):
        self.assertEqual(analysis.TOTAL_BATCH, 1024)
        self.assertEqual(analysis.LOCAL_ATTENTION_REQUESTS, 7)
        self.assertEqual(analysis.LOCAL_ROUTED_ASSIGNMENTS, 32)
        self.assertEqual(analysis.ATTENTION_TP_PAYLOAD_BYTES, 229376.0)
        self.assertEqual(analysis.ROUTED_TP_PAYLOAD_BYTES, 1048576.0)
        self.assertEqual(analysis.EP_DISPATCH_PAYLOAD_BYTES, 114688.0)
        self.assertEqual(analysis.EP_COMBINE_PAYLOAD_BYTES, 458752.0)

    def test_800_gbps_result_excludes_unconfigured_shared_experts(self):
        row = analysis.calculate_row(800)

        self.assertAlmostEqual(row.tp_ms, 2.60217856)
        self.assertAlmostEqual(row.ep_ms, 2.450176)
        self.assertAlmostEqual(row.total_ms, 27.80396056)
        self.assertAlmostEqual(row.user_tokens_per_s, 35.9660990686)
        self.assertAlmostEqual(row.system_tokens_per_s, 36829.2854462)
        self.assertEqual(
            row.paths,
            (
                "intra_node",
                "intra_node",
                "intra_node",
                "intra_node",
            ),
        )

    def test_scan_prints_eight_rows_without_creating_files(self):
        rows = analysis.scan_rows()
        self.assertEqual(
            [row.intra_node_gbps for row in rows],
            list(range(100, 801, 100)),
        )
        self.assertTrue(
            all(
                left.user_tokens_per_s < right.user_tokens_per_s
                for left, right in zip(rows, rows[1:])
            )
        )

        original_cwd = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                output = io.StringIO()
                with redirect_stdout(output):
                    analysis.main()
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                os.chdir(original_cwd)

        text = output.getvalue()
        self.assertIn("intra_GBps,tp_ms,ep_ms,total_ms", text)
        self.assertIn(
            "800,2.602179,2.450176,27.803961,35.966099,36829.285",
            text,
        )
        self.assertEqual(
            sum(line[:1].isdigit() for line in text.splitlines()),
            8,
        )

    def test_script_runs_directly_from_repository_root(self):
        repository = Path(__file__).resolve().parents[1]
        script = repository / "examples" / "analysis" / "scan_640_card_intra_node.py"

        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("local_attention_requests=7", completed.stdout)
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
