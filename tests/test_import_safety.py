import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class ImportSafetyTests(unittest.TestCase):
    def test_importing_entrypoint_modules_has_no_runtime_side_effects(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "must-not-be-created"
            environment = os.environ.copy()
            environment.update(
                {
                    "DATABASE_LOCATION": str(database_path),
                    "PYTHONPATH": str(repository_root),
                }
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "\n".join(
                        (
                            "import importlib",
                            "from unittest.mock import patch",
                            "import local_rag.ingestion",
                            "import local_rag.app",
                            "with patch('local_rag.app.render_app') as render_app:",
                            "    importlib.import_module('app')",
                            "    importlib.import_module('1_generate_embedding')",
                            "    importlib.import_module('2_start_chatbot')",
                            "    assert not render_app.called, 'import started Streamlit'",
                        )
                    ),
                ],
                cwd=temporary_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(database_path.exists())


if __name__ == "__main__":
    unittest.main()
