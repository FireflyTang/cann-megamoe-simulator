import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("case", SCRIPTS / "case.py")
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)


class CaseProvenance(unittest.TestCase):
    def test_git_identity_reports_commit_and_dirty_state(self):
        with tempfile.TemporaryDirectory() as name:
            repo = Path(name)
            env = {
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
                "PATH": "/usr/bin:/bin",
                "HOME": name,
            }
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "file").write_text("test\n")
            subprocess.run(["git", "add", "file"], check=True, cwd=repo, env=env)
            subprocess.run(["git", "commit", "-m", "test"], check=True, cwd=repo, env=env)
            identity = case.git_identity(repo)
            self.assertFalse(identity["dirty"])
            expected = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True,
                capture_output=True,
            ).stdout.strip()
            self.assertEqual(identity["commit"], expected)
            (repo / "dirty").write_text("dirty\n")
            self.assertTrue(case.git_identity(repo)["dirty"])


if __name__ == "__main__":
    unittest.main()
