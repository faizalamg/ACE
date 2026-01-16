#!/usr/bin/env python3
"""ACE Setup Wizard - One-shot installation and configuration script.

This script guides users through the complete ACE setup process:
1. Python version verification (3.11+)
2. ACE package installation from GitHub
3. Qdrant Docker container setup
4. API key configuration
5. .env file creation
6. Connection testing
7. MCP client configuration instructions

Usage:
    python setup_ace.py

Requirements:
    - Python 3.11+
    - Docker (for Qdrant)
    - Internet connection
"""

import os
import sys
import subprocess
import shutil
import platform
import time
from pathlib import Path
from typing import Optional, Tuple

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str) -> None:
    """Print a styled header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(step: int, total: int, text: str) -> None:
    """Print a step indicator."""
    print(f"{Colors.CYAN}[{step}/{total}]{Colors.ENDC} {Colors.BOLD}{text}{Colors.ENDC}")

def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {text}")

def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} {text}")

def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {text}")

def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {text}")

def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{question} [{default_str}]: ").strip().lower()
        if response == "":
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'")

def prompt_input(question: str, default: str = "", secret: bool = False) -> str:
    """Prompt user for input."""
    if default:
        prompt = f"{question} [{default}]: "
    else:
        prompt = f"{question}: "
    
    if secret:
        import getpass
        response = getpass.getpass(prompt)
    else:
        response = input(prompt).strip()
    
    return response if response else default

def run_command(cmd: list, capture: bool = True, check: bool = True) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except Exception as e:
        return -1, "", str(e)

def check_python_version() -> bool:
    """Check if Python version is 3.11+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.11+ required, found {version.major}.{version.minor}.{version.micro}")
        print_info("Please install Python 3.11+ from https://python.org")
        return False

def check_docker() -> bool:
    """Check if Docker is installed and running."""
    # Check if docker command exists
    returncode, stdout, stderr = run_command(["docker", "--version"])
    if returncode != 0:
        print_warning("Docker not found or not in PATH")
        return False
    
    print_success(f"Docker found: {stdout.strip()}")
    
    # Check if Docker daemon is running
    returncode, stdout, stderr = run_command(["docker", "info"])
    if returncode != 0:
        print_warning("Docker is installed but daemon is not running")
        print_info("Please start Docker Desktop or the Docker service")
        return False
    
    print_success("Docker daemon is running")
    return True

def check_qdrant_container() -> Optional[str]:
    """Check if Qdrant container exists and get its status."""
    returncode, stdout, stderr = run_command([
        "docker", "ps", "-a", 
        "--filter", "name=qdrant",
        "--format", "{{.Status}}"
    ])
    if returncode == 0 and stdout.strip():
        return stdout.strip()
    return None

def start_qdrant() -> bool:
    """Start or create Qdrant container."""
    status = check_qdrant_container()
    
    if status and "Up" in status:
        print_success("Qdrant container is already running")
        return True
    
    if status and "Exited" in status:
        print_info("Starting existing Qdrant container...")
        returncode, _, stderr = run_command(["docker", "start", "qdrant"])
        if returncode == 0:
            print_success("Qdrant container started")
            return True
        else:
            print_error(f"Failed to start Qdrant: {stderr}")
            return False
    
    # Create new container
    print_info("Creating new Qdrant container...")
    returncode, _, stderr = run_command([
        "docker", "run", "-d",
        "--name", "qdrant",
        "-p", "6333:6333",
        "-p", "6334:6334",
        "-v", "qdrant_storage:/qdrant/storage",
        "qdrant/qdrant"
    ])
    
    if returncode == 0:
        print_success("Qdrant container created and started")
        print_info("Waiting for Qdrant to initialize...")
        time.sleep(3)  # Give Qdrant time to start
        return True
    else:
        print_error(f"Failed to create Qdrant container: {stderr}")
        return False

def test_qdrant_connection() -> bool:
    """Test connection to Qdrant."""
    try:
        import httpx
        response = httpx.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print_success("Qdrant connection verified")
            return True
    except ImportError:
        # Try with urllib if httpx not installed
        import urllib.request
        try:
            with urllib.request.urlopen("http://localhost:6333/collections", timeout=5) as response:
                if response.status == 200:
                    print_success("Qdrant connection verified")
                    return True
        except Exception:
            pass
    except Exception:
        pass
    
    print_warning("Could not connect to Qdrant at localhost:6333")
    return False

def install_ace() -> bool:
    """Install ACE from GitHub."""
    print_info("Installing ACE from GitHub...")
    
    returncode, stdout, stderr = run_command([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/erwinh22/ACE.git",
        "--upgrade"
    ], capture=False)
    
    if returncode == 0:
        print_success("ACE installed successfully")
        return True
    else:
        print_error("Failed to install ACE")
        print_info("Try manually: pip install git+https://github.com/erwinh22/ACE.git")
        return False

def create_env_file(env_vars: dict) -> bool:
    """Create or update .env file with API keys."""
    env_file = Path.cwd() / ".env"
    
    # Read existing .env if present
    existing_vars = {}
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    existing_vars[key.strip()] = value.strip()
    
    # Merge with new vars (new takes precedence)
    merged_vars = {**existing_vars, **env_vars}
    
    # Write .env file
    try:
        with open(env_file, "w") as f:
            f.write("# ACE Configuration\n")
            f.write("# Generated by setup_ace.py\n\n")
            
            f.write("# LLM Provider (choose one)\n")
            if "ZAI_API_KEY" in merged_vars:
                f.write(f"ZAI_API_KEY={merged_vars['ZAI_API_KEY']}\n")
            if "OPENAI_API_KEY" in merged_vars:
                f.write(f"OPENAI_API_KEY={merged_vars['OPENAI_API_KEY']}\n")
            
            f.write("\n# Code Indexing (optional but recommended for 94% retrieval accuracy)\n")
            if "VOYAGE_API_KEY" in merged_vars:
                f.write(f"VOYAGE_API_KEY={merged_vars['VOYAGE_API_KEY']}\n")
            
            f.write("\n# Observability (optional)\n")
            if "OPIK_API_KEY" in merged_vars:
                f.write(f"OPIK_API_KEY={merged_vars['OPIK_API_KEY']}\n")
            
            f.write("\n# Qdrant Configuration\n")
            f.write(f"ACE_QDRANT_URL={merged_vars.get('ACE_QDRANT_URL', 'http://localhost:6333')}\n")
        
        print_success(f"Created {env_file}")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False

def update_gitignore() -> None:
    """Ensure .env is in .gitignore."""
    gitignore = Path.cwd() / ".gitignore"
    
    if gitignore.exists():
        with open(gitignore, "r") as f:
            content = f.read()
        if ".env" not in content:
            with open(gitignore, "a") as f:
                f.write("\n# ACE secrets\n.env\n")
            print_info("Added .env to .gitignore")
    else:
        with open(gitignore, "w") as f:
            f.write("# ACE secrets\n.env\n")
        print_info("Created .gitignore with .env")

def print_mcp_config() -> None:
    """Print MCP client configuration instructions."""
    ace_path = Path(__file__).parent / "ace_mcp_server.py"
    python_path = sys.executable
    
    print_header("MCP Client Configuration")
    
    print(f"""
{Colors.BOLD}VS Code Copilot / GitHub Copilot Chat:{Colors.ENDC}

Add to your VS Code settings.json (User or Workspace):

{Colors.CYAN}"mcp": {{
  "servers": {{
    "ace": {{
      "command": "{python_path.replace(chr(92), '/')}",
      "args": ["{str(ace_path).replace(chr(92), '/')}"]
    }}
  }}
}}{Colors.ENDC}

{Colors.BOLD}Claude Desktop:{Colors.ENDC}

Add to claude_desktop_config.json:

{Colors.CYAN}{{
  "mcpServers": {{
    "ace": {{
      "command": "{python_path.replace(chr(92), '/')}",
      "args": ["{str(ace_path).replace(chr(92), '/')}"]
    }}
  }}
}}{Colors.ENDC}

{Colors.BOLD}Cursor:{Colors.ENDC}

Add to .cursor/mcp.json in your project:

{Colors.CYAN}{{
  "servers": {{
    "ace": {{
      "command": "{python_path.replace(chr(92), '/')}",
      "args": ["{str(ace_path).replace(chr(92), '/')}"]
    }}
  }}
}}{Colors.ENDC}
""")

def main():
    """Main setup wizard."""
    print_header("ACE Setup Wizard v0.8.0")
    print("This wizard will guide you through ACE installation and configuration.\n")
    
    total_steps = 6
    env_vars = {}
    
    # Step 1: Python version
    print_step(1, total_steps, "Checking Python version...")
    if not check_python_version():
        print_error("Setup cannot continue without Python 3.11+")
        sys.exit(1)
    
    # Step 2: Docker and Qdrant
    print_step(2, total_steps, "Setting up Qdrant (vector database)...")
    docker_ok = check_docker()
    qdrant_ok = False
    
    if docker_ok:
        if prompt_yes_no("Start Qdrant container?"):
            qdrant_ok = start_qdrant()
            if qdrant_ok:
                # Wait a bit more and test connection
                time.sleep(2)
                test_qdrant_connection()
    else:
        print_warning("Docker not available. You'll need to set up Qdrant manually.")
        print_info("Option 1: Install Docker Desktop and re-run this script")
        print_info("Option 2: Use Qdrant Cloud: https://cloud.qdrant.io")
        print_info("Option 3: Download Qdrant binary: https://qdrant.tech/documentation/quick-start/")
        
        qdrant_url = prompt_input("Enter Qdrant URL", "http://localhost:6333")
        env_vars["ACE_QDRANT_URL"] = qdrant_url
    
    # Step 3: Install ACE
    print_step(3, total_steps, "Installing ACE package...")
    if prompt_yes_no("Install ACE from GitHub?"):
        install_ace()
    else:
        print_info("Skipping ACE installation")
        print_info("Install manually: pip install git+https://github.com/erwinh22/ACE.git")
    
    # Step 4: API Keys
    print_step(4, total_steps, "Configuring API keys...")
    print()
    print(f"{Colors.BOLD}LLM Provider (required - choose one):{Colors.ENDC}")
    print("  1. Z.ai (GLM models) - https://z.ai")
    print("  2. OpenAI - https://platform.openai.com")
    print("  3. Skip for now")
    
    llm_choice = prompt_input("Select LLM provider [1/2/3]", "1")
    
    if llm_choice == "1":
        api_key = prompt_input("Enter ZAI_API_KEY", secret=True)
        if api_key:
            env_vars["ZAI_API_KEY"] = api_key
    elif llm_choice == "2":
        api_key = prompt_input("Enter OPENAI_API_KEY", secret=True)
        if api_key:
            env_vars["OPENAI_API_KEY"] = api_key
    
    print()
    print(f"{Colors.BOLD}Code Indexing (optional but recommended):{Colors.ENDC}")
    print("  Voyage AI provides 94% retrieval accuracy for code")
    print("  New accounts get 200M FREE tokens: https://voyageai.com")
    
    if prompt_yes_no("Configure Voyage AI for code indexing?", default=False):
        voyage_key = prompt_input("Enter VOYAGE_API_KEY", secret=True)
        if voyage_key:
            env_vars["VOYAGE_API_KEY"] = voyage_key
    
    print()
    print(f"{Colors.BOLD}Observability (optional):{Colors.ENDC}")
    print("  Opik provides metrics and monitoring: https://comet.com/site/products/opik")
    
    if prompt_yes_no("Configure Opik for observability?", default=False):
        opik_key = prompt_input("Enter OPIK_API_KEY", secret=True)
        if opik_key:
            env_vars["OPIK_API_KEY"] = opik_key
    
    # Step 5: Create .env
    print_step(5, total_steps, "Creating configuration files...")
    if env_vars:
        create_env_file(env_vars)
        update_gitignore()
    else:
        print_warning("No API keys configured. Create .env manually when ready.")
    
    # Step 6: MCP Configuration
    print_step(6, total_steps, "MCP Client Setup...")
    print_mcp_config()
    
    # Summary
    print_header("Setup Complete!")
    print(f"""
{Colors.BOLD}Summary:{Colors.ENDC}
  - Python: {Colors.GREEN}OK{Colors.ENDC}
  - Qdrant: {Colors.GREEN if qdrant_ok else Colors.YELLOW}{'Running' if qdrant_ok else 'Manual setup needed'}{Colors.ENDC}
  - ACE Package: Check with 'python -c "import ace; print(ace.__version__)"'
  - API Keys: {'Configured in .env' if env_vars else 'Not configured'}

{Colors.BOLD}Next Steps:{Colors.ENDC}
  1. Add MCP configuration to your editor (see above)
  2. Restart your editor/IDE
  3. Start using ACE! The first 'ace_retrieve' call will auto-onboard your workspace

{Colors.BOLD}Quick Test:{Colors.ENDC}
  python -c "from ace.unified_memory import UnifiedMemoryIndex; print('ACE ready!')"

{Colors.BOLD}Documentation:{Colors.ENDC}
  - README: https://github.com/erwinh22/ACE/blob/main/README.md
  - MCP Guide: https://github.com/erwinh22/ACE/blob/main/README_MCPAgent.md

{Colors.YELLOW}Need help? Open an issue: https://github.com/erwinh22/ACE/issues{Colors.ENDC}
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup cancelled by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
