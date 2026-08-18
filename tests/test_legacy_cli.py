from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_legacy(*arguments):
    return subprocess.run(
        [sys.executable, "main.py", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )


class LegacyCliTests(unittest.TestCase):
    def test_help_retains_config_path(self):
        result = run_legacy("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config-path", result.stdout)

    def test_qwen3_decode_command_retains_tpot_output(self):
        result = run_legacy(
            "--config-path",
            "hf_configs/qwen3-8B_config.json",
            "--device-type",
            "H20",
            "--world-size",
            "1",
            "--tp-size",
            "1",
            "--decode-only",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TPOT (ms)", result.stdout)


if __name__ == "__main__":
    unittest.main()
