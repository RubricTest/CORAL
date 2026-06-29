#!/usr/bin/env python3
"""Probe: inside a task container, what tooling is available and can we install
mini-swe-agent? (No model needed — just feasibility of the agent install.)"""
import asyncio, sys
from env_explore import EnvSandbox

IMAGE = sys.argv[1] if len(sys.argv) > 1 else (
    "namanjain12/aiohttp_final:006fbe03fede4eaa1eeba7b8393cbf4d63cb44b6"
)

PROBES = [
    ("python/pip/uv/pipx present", "python --version; python3 --version; which pip pip3 uv pipx 2>/dev/null; echo ---"),
    ("internet -> pypi", "python - <<'PY'\nimport urllib.request,socket\ntry:\n  socket.setdefaulttimeout(8); urllib.request.urlopen('https://pypi.org/simple/mini-swe-agent/'); print('PYPI REACHABLE')\nexcept Exception as e: print('PYPI FAIL', e)\nPY"),
    ("uv tool install mini-swe-agent (preferred, own python)", "command -v uv >/dev/null && uv tool install mini-swe-agent 2>&1 | tail -5 || echo 'no uv'"),
    ("pip install mini-swe-agent (fallback)", "pip install -q mini-swe-agent 2>&1 | tail -8; echo rc=$?"),
    ("mini-swe-agent on PATH?", "export PATH=$PATH:$HOME/.local/bin; command -v mini-swe-agent mini && (mini-swe-agent --help 2>&1 | head -3 || mini --help 2>&1 | head -3) || echo 'NOT FOUND'"),
]


async def main():
    async with EnvSandbox(IMAGE) as env:
        print(f"container ready id={env._sandbox.id}\n", flush=True)
        for name, cmd in PROBES:
            rc, out, err = await env.exec(cmd, timeout_sec=300)
            print(f"### {name}  (rc={rc})\n{(out or '').strip()[:1200]}", flush=True)
            if (err or "").strip():
                print(f"  [stderr] {err.strip()[:400]}", flush=True)
            print(flush=True)

if __name__ == "__main__":
    asyncio.run(main())
