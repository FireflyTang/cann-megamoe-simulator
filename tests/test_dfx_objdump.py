import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("dfx", SCRIPTS / "dfx.py")
dfx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfx)


class DfxObjdump(unittest.TestCase):
    def test_environment_selects_explicit_objdump(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "llvm-objdump"
            path.write_text("#!/bin/sh\n")
            path.chmod(0o755)
            old = os.environ.get("MEGAMOE_LLVM_OBJDUMP")
            os.environ["MEGAMOE_LLVM_OBJDUMP"] = str(path)
            try:
                self.assertEqual(dfx.find_llvm_objdump(), str(path))
            finally:
                if old is None:
                    os.environ.pop("MEGAMOE_LLVM_OBJDUMP", None)
                else:
                    os.environ["MEGAMOE_LLVM_OBJDUMP"] = old


if __name__ == "__main__":
    unittest.main()
