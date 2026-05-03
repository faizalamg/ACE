"""Auto-launcher for llama.cpp server (Gemma 4 26B).

Checks if the llama-server is running on the configured port and starts it if not.
Called by LLM initialization code before making inference requests.
"""
import logging
import os
import subprocess
import time

import httpx

logger = logging.getLogger(__name__)

# Default llama-server configuration
LLAMA_SERVER_EXE = os.environ.get(
    "ACE_LLAMA_SERVER_EXE",
    r"C:\Users\Erwin\llama-cpp-hip\llama-server.exe",
)
LLAMA_MODEL_PATH = os.environ.get(
    "ACE_LLAMA_MODEL_PATH",
    r"C:\ollama\models\WizardCoder-15B-V1.0\unsloth\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-UD-Q3_K_M.gguf",
)
LLAMA_PORT = int(os.environ.get("ACE_LLAMA_PORT", "8091"))


def _parse_port_from_url(url: str) -> int:
    """Extract port number from a URL like http://localhost:8091/v1."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.port or LLAMA_PORT
    except Exception:
        return LLAMA_PORT


def is_server_running(port: int = None, timeout: float = 3.0) -> bool:
    """Check if llama-server is responding on the given port."""
    port = port or LLAMA_PORT
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"http://localhost:{port}/health")
            return resp.status_code == 200
    except Exception:
        return False


def ensure_server_running(llm_url: str = None, timeout: float = 90.0) -> bool:
    """Ensure llama-server is running. Start it if not.

    Args:
        llm_url: The LLM URL from config (e.g. http://localhost:8091/v1).
                 Port is extracted from this URL.
        timeout: Max seconds to wait for server to become ready after starting.

    Returns:
        True if server is running (was already running or successfully started).
        False if server could not be started.
    """
    port = _parse_port_from_url(llm_url) if llm_url else LLAMA_PORT

    # Already running?
    if is_server_running(port):
        logger.debug(f"llama-server already running on port {port}")
        return True

    # Check binary exists
    exe = LLAMA_SERVER_EXE
    if not os.path.isfile(exe):
        logger.error(f"llama-server not found at {exe}. Set ACE_LLAMA_SERVER_EXE env var.")
        return False

    # Check model exists
    model = LLAMA_MODEL_PATH
    if not os.path.isfile(model):
        logger.error(f"Model not found at {model}. Set ACE_LLAMA_MODEL_PATH env var.")
        return False

    # Launch with optimized settings
    cmd = [
        exe,
        "-m", model,
        "--port", str(port),
        "-ngl", "99",
        "-t", "16",
        "-tb", "32",
        "-fa", "on",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-c", "65536",
        "-np", "1",
        "--mlock",
        "--sleep-idle-seconds", "120",
        "--metrics",
    ]

    logger.info(f"Starting llama-server on port {port}: {' '.join(cmd[:6])}...")

    try:
        # Start as detached background process
        if os.name == "nt":
            # Windows: CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
            )
        else:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as e:
        logger.error(f"Failed to start llama-server: {e}")
        return False

    # Wait for server to become ready
    start = time.monotonic()
    poll_interval = 2.0
    while time.monotonic() - start < timeout:
        if is_server_running(port):
            elapsed = time.monotonic() - start
            logger.info(f"llama-server ready on port {port} (took {elapsed:.1f}s)")
            return True
        time.sleep(poll_interval)

    logger.error(f"llama-server failed to start within {timeout}s on port {port}")
    return False
