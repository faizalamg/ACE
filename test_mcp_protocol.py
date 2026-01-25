#!/usr/bin/env python
"""Test ACE MCP server via actual JSON-RPC protocol with bidirectional handling."""
import subprocess
import time
import json
import sys
import threading
import os

# Set env
os.environ["QDRANT_URL"] = "http://localhost:6333"

WORKSPACE_PATH = r"D:\ApplicationDevelopment\Tools\agentic-context-engine"

print("=== MCP PROTOCOL TEST (with roots/list handling) ===", flush=True)

# Start MCP server
proc = subprocess.Popen(
    [".venv/Scripts/python.exe", "ace_mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,  # line buffered
)

# Read stderr in background thread
def read_stderr():
    try:
        for line in iter(proc.stderr.readline, ""):
            if line:
                print(f"[STDERR] {line.rstrip()}", flush=True)
    except:
        pass

t = threading.Thread(target=read_stderr, daemon=True)
t.start()

# Give server a moment to start
time.sleep(1)

# Send initialize - MUST declare roots capability
init_req = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "roots": {"listChanged": True}  # Tell server we support roots/list
        },
        "clientInfo": {"name": "test-client", "version": "1.0"},
    },
    "id": 1,
}
print(f">>> Sending initialize (with roots capability)...", flush=True)
proc.stdin.write(json.dumps(init_req) + "\n")
proc.stdin.flush()

# Read init response
print("<<< Waiting for init response...", flush=True)
init_resp = proc.stdout.readline()
print(f"<<< Init: {init_resp[:200].strip()}...", flush=True)

# Send initialized notification
notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
proc.stdin.write(json.dumps(notif) + "\n")
proc.stdin.flush()
print(">>> Sent initialized notification", flush=True)

time.sleep(0.5)

# Send tool call
tool_call = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "ace_retrieve", "arguments": {"query": "debugging test", "limit": 2}},
    "id": 2,
}
print(f"\n>>> Sending ace_retrieve at t=0...", flush=True)
start = time.time()
proc.stdin.write(json.dumps(tool_call) + "\n")
proc.stdin.flush()

# Wait for response (up to 60s) - handle server requests!
print("<<< Waiting for response (max 60s, handling server requests)...", flush=True)
timeout = 60

while True:
    elapsed = time.time() - start
    if elapsed > timeout:
        print(f"\n!!! TIMEOUT after {timeout}s", flush=True)
        break
    
    # Check if process died
    if proc.poll() is not None:
        print(f"\n!!! Process exited with code {proc.returncode}", flush=True)
        break
    
    # Try to read stdout
    try:
        line = proc.stdout.readline()
        if line:
            print(f"<<< {elapsed:.1f}s: {line.strip()[:300]}", flush=True)
            try:
                msg = json.loads(line)
                
                # Check if this is a REQUEST from server (has method but also has id)
                if "method" in msg and "id" in msg:
                    method = msg["method"]
                    req_id = msg["id"]
                    print(f"    [SERVER REQUEST: {method}, id={req_id}]", flush=True)
                    
                    if method == "roots/list":
                        # Respond with workspace roots
                        response = {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "roots": [
                                    {"uri": f"file:///{WORKSPACE_PATH.replace(chr(92), '/')}", "name": "workspace"}
                                ]
                            }
                        }
                        print(f">>> Responding to roots/list with: {WORKSPACE_PATH}", flush=True)
                        proc.stdin.write(json.dumps(response) + "\n")
                        proc.stdin.flush()
                    else:
                        # Unknown request - send empty result
                        response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
                        proc.stdin.write(json.dumps(response) + "\n")
                        proc.stdin.flush()
                    continue
                
                # Check if this is a RESPONSE to our request
                if "result" in msg or "error" in msg:
                    print(f"\n=== SUCCESS - Response in {elapsed:.1f}s ===", flush=True)
                    if "error" in msg:
                        print(f"ERROR: {msg['error']}", flush=True)
                    else:
                        result = msg.get("result", {})
                        content = result.get("content", [])
                        print(f"Content items: {len(content)}", flush=True)
                        if content:
                            text = content[0].get("text", "")[:500]
                            print(f"First item preview: {text}...", flush=True)
                    break
                    
            except json.JSONDecodeError:
                pass  # Not JSON, keep waiting
    except Exception as e:
        print(f"Read error: {e}", flush=True)
        time.sleep(0.1)

# Cleanup
print("\nKilling server...", flush=True)
proc.kill()
proc.wait()
print("Done.", flush=True)
