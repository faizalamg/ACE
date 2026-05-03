import os
from unittest.mock import MagicMock, patch


def test_spawn_daemon_forces_local_lmstudio_indexing(tmp_path):
    """Watcher daemon should force local LM Studio indexing env."""
    from ace.file_watcher_daemon import _spawn_daemon

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    process = MagicMock(pid=12345)

    with patch("ace.file_watcher_daemon.is_watcher_running", side_effect=[False, True]), \
         patch("ace.file_watcher_daemon.time.sleep"), \
         patch("ace.file_watcher_daemon.subprocess.Popen", return_value=process) as mock_popen:
        result = _spawn_daemon(str(workspace), "http://localhost:6333")

    assert result == 0
    kwargs = mock_popen.call_args.kwargs
    env = kwargs["env"]
    assert env["ACE_TEXT_EMBEDDING_PROVIDER"] == "local"
    assert env["ACE_CODE_EMBEDDING_PROVIDER"] == "local"
    assert env["ACE_LOCAL_EMBEDDING_URL"] == os.environ.get("ACE_LOCAL_EMBEDDING_URL", "http://localhost:1234")
