"""Offline app checks with mocked HF OAuth, never a real login or label store."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class SpaceBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gradio.oauth
        except ImportError:
            raise unittest.SkipTest("Install space/requirements.txt for app boundary tests")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))
        with patch("gradio.oauth._get_mocked_oauth_info", return_value={"userinfo": {"preferred_username": "synthetic-user", "sub": "test"}}):
            cls.app = importlib.import_module("app")

    def test_unauthenticated_upload_refused(self):
        import gradio as gr
        with self.assertRaises(gr.Error):
            self.app.submit("hstu", None, None)

    def test_private_path_not_accepted_as_upload(self):
        import gradio as gr
        from types import SimpleNamespace
        with patch.object(self.app, "READY", True):
            with self.assertRaises(gr.Error):
                self.app.submit("hstu", "/tmp/autorse-private/hstu/beauty/test_targets.npy", SimpleNamespace(username="test"))

    def test_blocked_private_file_is_not_served_over_http(self):
        from fastapi.testclient import TestClient
        from gradio.routes import App
        with tempfile.TemporaryDirectory() as temp:
            label = Path(temp) / "synthetic-private-labels.txt"
            label.write_text("synthetic fixture only")
            with patch.object(self.app.demo, "allowed_paths", [temp]), patch.object(self.app.demo, "blocked_paths", [temp]), \
                    patch("gradio.oauth._get_mocked_oauth_info", return_value={"userinfo": {"sub": "test"}}):
                with TestClient(App.create_app(self.app.demo)) as client:
                    self.assertEqual(client.get("/gradio_api/file=" + str(label)).status_code, 403)

    def test_daily_limit_is_atomic_and_task_scoped(self):
        from space.service import SubmissionRejected
        # app imports service with a script-relative module name.
        import service
        with tempfile.TemporaryDirectory() as temp, patch.object(self.app, "STATE_ROOT", Path(temp)), patch.object(self.app, "DAILY_LIMIT", 2):
            self.app.reserve_attempt("user", "hstu")
            self.app.reserve_attempt("user", "hstu")
            with self.assertRaises(service.SubmissionRejected):
                self.app.reserve_attempt("user", "hstu")
            self.app.reserve_attempt("user", "tiger")


if __name__ == "__main__":
    unittest.main()
