import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("export_msprof", SCRIPTS / "export_msprof.py")
export_msprof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_msprof)


class ExportMsprofCommand(unittest.TestCase):
    def test_msprof_arguments_use_equals_form(self):
        command = export_msprof.build_command(
            Path("/tmp/dump"), Path("/tmp/output"), "dav_3510", "all", False
        )
        self.assertEqual(command, [
            "msprof", "op", "simulator",
            "--export=/tmp/dump", "--output=/tmp/output",
            "--soc-version=dav_3510", "--aic-metrics=PipeUtilization",
        ])

    def test_explicit_core_id_uses_equals_form(self):
        command = export_msprof.build_command(
            Path("/tmp/dump"), Path("/tmp/output"), "dav_3510", "0,1", True
        )
        self.assertIn("--core-id=0,1", command)


if __name__ == "__main__":
    unittest.main()
