#!/usr/bin/env python3
"""ACE File Watcher Daemon - Persistent background file watcher.

This daemon runs as a background process and monitors workspace files
for changes, automatically reindexing when code is modified.

Usage:
    python -m ace.file_watcher_daemon start /path/to/workspace
    python -m ace.file_watcher_daemon stop /path/to/workspace
    python -m ace.file_watcher_daemon status /path/to/workspace

The daemon stores its PID in .ace/.watcher.pid and logs to .ace/.watcher.log
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add ace package to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace.code_indexer import CodeIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Daemon configuration
_ACE_DIR = ".ace"
_PID_FILE = ".watcher.pid"
_LOG_FILE = ".watcher.log"
_WATCHED_DIRS_FILE = ".watched_workspaces"


def get_workspace_ace_dir(workspace_path: str) -> Path:
    """Get the .ace directory for a workspace."""
    return Path(workspace_path) / _ACE_DIR


def get_pid_file(workspace_path: str) -> Path:
    """Get the PID file path for a workspace watcher."""
    return get_workspace_ace_dir(workspace_path) / _PID_FILE


def get_log_file(workspace_path: str) -> Path:
    """Get the log file path for a workspace watcher."""
    return get_workspace_ace_dir(workspace_path) / _LOG_FILE


def get_watched_workspaces_file() -> Path:
    """Get the file tracking all watched workspaces."""
    ace_dir = Path.home() / ".claude" / ".ace"
    ace_dir.mkdir(parents=True, exist_ok=True)
    return ace_dir / _WATCHED_DIRS_FILE


def is_watcher_running(workspace_path: str) -> bool:
    """Check if a watcher daemon is running for the workspace."""
    pid_file = get_pid_file(workspace_path)

    if not pid_file.exists():
        return False

    try:
        pid = int(pid_file.read_text().strip())
        # Check if process is running
        # Windows: os.kill(pid, 0) doesn't work properly, use alternative
        if os.name == 'nt':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            process = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if process:
                kernel32.CloseHandle(process)
                return True
            else:
                # Process doesn't exist, clean up stale PID file
                pid_file.unlink(missing_ok=True)
                return False
        else:
            # Unix: use signal 0 to check process existence
            try:
                os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
                return True
            except OSError:
                # Process doesn't exist, clean up stale PID file
                pid_file.unlink(missing_ok=True)
                return False
    except (ValueError, IOError):
        return False


def write_pid_file(workspace_path: str) -> None:
    """Write the current process PID to the PID file."""
    pid_file = get_pid_file(workspace_path)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    logger.info(f"Wrote PID file: {pid_file} (PID: {os.getpid()})")


def remove_pid_file(workspace_path: str) -> None:
    """Remove the PID file for a workspace."""
    get_pid_file(workspace_path).unlink(missing_ok=True)


def register_workspace(workspace_path: str) -> None:
    """Register a workspace as being watched."""
    watched_file = get_watched_workspaces_file()
    workspaces = set()

    if watched_file.exists():
        try:
            for line in watched_file.read_text().strip().split('\n'):
                if line:
                    workspaces.add(line)
        except IOError:
            pass

    workspaces.add(str(Path(workspace_path).resolve()))

    with open(watched_file, 'w') as f:
        f.write('\n'.join(workspaces))


def unregister_workspace(workspace_path: str) -> None:
    """Unregister a workspace from being watched."""
    watched_file = get_watched_workspaces_file()

    if not watched_file.exists():
        return

    try:
        workspaces = set()
        for line in watched_file.read_text().strip().split('\n'):
            if line and line != str(Path(workspace_path).resolve()):
                workspaces.add(line)

        if workspaces:
            with open(watched_file, 'w') as f:
                f.write('\n'.join(workspaces))
        else:
            watched_file.unlink(missing_ok=True)
    except IOError:
        pass


def start_watcher(workspace_path: str, qdrant_url: str = "http://localhost:6333") -> int:
    """Start the file watcher daemon for a workspace.

    Returns:
        0 if already running or started successfully
        1 if failed to start
    """
    workspace_path = str(Path(workspace_path).resolve())

    # Check if already running
    if is_watcher_running(workspace_path):
        print(f"File watcher already running for: {workspace_path}")
        return 0

    # Check if workspace exists
    if not os.path.isdir(workspace_path):
        print(f"Error: Workspace directory does not exist: {workspace_path}")
        return 1

    # Setup logging for this workspace
    log_file = get_log_file(workspace_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure file handler for workspace-specific logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Starting file watcher for workspace: {workspace_path}")

    # Write PID file
    write_pid_file(workspace_path)
    atexit.register(remove_pid_file, workspace_path)

    # Register workspace
    register_workspace(workspace_path)

    # Get collection name from workspace config
    collection_name = None
    config_file = get_workspace_ace_dir(workspace_path) / ".ace.json"
    if config_file.exists():
        try:
            import json
            config = json.loads(config_file.read_text())
            collection_name = config.get("collection_name")
        except (json.JSONDecodeError, IOError):
            pass

    if not collection_name:
        # Derive from folder name
        workspace_name = os.path.basename(os.path.normpath(workspace_path))
        collection_name = f"{workspace_name}_code_context"

    # Create indexer
    indexer = CodeIndexer(
        workspace_path=workspace_path,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
    )

    # Start watching
    try:
        indexer.start_watching()
        print(f"File watcher started for: {workspace_path}")
        print(f"  Collection: {collection_name}")
        print(f"  PID: {os.getpid()}")
        print(f"  Log: {log_file}")

        # Keep the daemon running
        # The watcher runs in a background thread
        logger.info("Entering daemon keep-alive loop")
        while True:
            time.sleep(60)
            logger.debug("Daemon heartbeat")

    except KeyboardInterrupt:
        logger.info("File watcher stopped by user")
        indexer.stop_watching()
        remove_pid_file(workspace_path)
        unregister_workspace(workspace_path)
        return 0
    except Exception as e:
        logger.exception(f"File watcher error: {e}")
        remove_pid_file(workspace_path)
        unregister_workspace(workspace_path)
        return 1


def stop_watcher(workspace_path: str) -> int:
    """Stop the file watcher daemon for a workspace.

    Returns:
        0 if stopped successfully or not running
        1 if failed to stop
    """
    workspace_path = str(Path(workspace_path).resolve())
    pid_file = get_pid_file(workspace_path)

    if not pid_file.exists():
        print(f"No file watcher running for: {workspace_path}")
        return 0

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 15)  # SIGTERM
        print(f"Stopped file watcher (PID {pid}) for: {workspace_path}")

        # Unregister workspace
        unregister_workspace(workspace_path)

        return 0
    except (ValueError, OSError) as e:
        print(f"Failed to stop file watcher: {e}")
        # Clean up stale PID file
        pid_file.unlink(missing_ok=True)
        return 1


def status_watcher(workspace_path: str) -> int:
    """Check the status of the file watcher for a workspace.

    Returns:
        0 if running, 1 if not running
    """
    workspace_path = str(Path(workspace_path).resolve())

    if is_watcher_running(workspace_path):
        pid_file = get_pid_file(workspace_path)
        pid = int(pid_file.read_text().strip())
        log_file = get_log_file(workspace_path)

        print(f"File watcher is RUNNING for: {workspace_path}")
        print(f"  PID: {pid}")
        print(f"  Log: {log_file}")
        return 0
    else:
        print(f"File watcher is NOT running for: {workspace_path}")
        return 1


def list_watched() -> int:
    """List all workspaces with active file watchers.

    Returns:
        0
    """
    watched_file = get_watched_workspaces_file()

    if not watched_file.exists():
        print("No workspaces are currently being watched.")
        return 0

    print("Watched workspaces:")
    running_count = 0

    for line in watched_file.read_text().strip().split('\n'):
        if line:
            if is_watcher_running(line):
                print(f"  [RUNNING] {line}")
                running_count += 1
            else:
                print(f"  [STOPPED] {line}")

    print(f"\nTotal: {running_count} running")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ACE File Watcher Daemon - Manage background file watchers"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start file watcher for a workspace')
    start_parser.add_argument('workspace', help='Path to the workspace directory')
    start_parser.add_argument('--qdrant-url', default='http://localhost:6333',
                             help='Qdrant server URL')

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop file watcher for a workspace')
    stop_parser.add_argument('workspace', help='Path to the workspace directory')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check file watcher status')
    status_parser.add_argument('workspace', help='Path to the workspace directory')

    # List command
    subparsers.add_parser('list', help='List all watched workspaces')

    # Hidden daemon-run mode (for internal use - runs the actual watcher)
    daemon_parser = subparsers.add_parser('_run_daemon', add_help=False)
    daemon_parser.add_argument('workspace', help='Path to the workspace directory')
    daemon_parser.add_argument('--qdrant-url', default='http://localhost:6333',
                              help='Qdrant server URL')

    args = parser.parse_args()

    if args.command == 'start':
        return _spawn_daemon(args.workspace, args.qdrant_url)
    elif args.command == '_run_daemon':
        return _run_daemon(args.workspace, args.qdrant_url)
    elif args.command == 'stop':
        return stop_watcher(args.workspace)
    elif args.command == 'status':
        return status_watcher(args.workspace)
    elif args.command == 'list':
        return list_watched()
    else:
        parser.print_help()
        return 1


def _spawn_daemon(workspace_path: str, qdrant_url: str) -> int:
    """Spawn the watcher as a detached daemon process.

    This is called when the script is invoked with 'start' command.
    It spawns a new Python process that runs the actual watcher.
    """
    import subprocess

    workspace_path = str(Path(workspace_path).resolve())

    # Check if already running
    if is_watcher_running(workspace_path):
        print(f"File watcher already running for: {workspace_path}")
        return 0

    # Get the path to this script
    script_path = Path(__file__).resolve().parent / "file_watcher_daemon.py"

    # Find pythonw.exe for Windows (background Python without console)
    if os.name == 'nt':
        python_exe = sys.executable.replace("python.exe", "pythonw.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable  # Fall back to regular python

        # Windows: use DETACHED_PROCESS to fully detach from parent
        # DETACHED_PROCESS = 0x00000008
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000

        creation_flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

        process = subprocess.Popen(
            [python_exe, str(script_path), '_run_daemon', workspace_path, '--qdrant-url', qdrant_url],
            creationflags=creation_flags,
            close_fds=True,
        )
    else:
        # Unix: use double fork to daemonize
        process = subprocess.Popen(
            [sys.executable, str(script_path), '_run_daemon', workspace_path, '--qdrant-url', qdrant_url],
            start_new_session=True,  # Creates new process group
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Give the daemon a moment to start and write PID
    time.sleep(2)

    # Check if it started successfully
    if is_watcher_running(workspace_path):
        print(f"File watcher started for: {workspace_path}")
        print(f"  PID: {process.pid}")
        log_file = get_log_file(workspace_path)
        print(f"  Log: {log_file}")
        return 0
    else:
        print(f"Failed to start file watcher for: {workspace_path}")
        print(f"  Check log: {get_log_file(workspace_path)}")
        return 1


def _run_daemon(workspace_path: str, qdrant_url: str) -> int:
    """Run the actual watcher daemon.

    This is called when the script is invoked with '--daemon' flag.
    It should NOT be called directly by users.
    """
    return start_watcher(workspace_path, qdrant_url)


if __name__ == '__main__':
    sys.exit(main())
