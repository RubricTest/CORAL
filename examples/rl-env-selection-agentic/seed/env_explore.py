#!/usr/bin/env python3
"""Spin up a task's RL environment (docker container) via OpenSandbox so the agent
can explore it — run commands, read files, optionally rollout the base model.

Each environment in this benchmark maps to a docker image
``namanjain12/<repo>_final:<commit_hash>`` (the SWE testbed at the buggy commit,
checked out at /testbed). This helper wraps the OpenSandbox SDK to:
    start(image) -> exec(cmd) / read_file(path) -> stop()

IMPORTANT: the task images live in the INTERNAL registry, not Docker Hub. The
``docker_image`` in task_env_inputs.jsonl is ``namanjain12/<repo>_final:<commit>``
(Docker-Hub style) but the OpenSandbox server resolves it from
``10.10.110.20:5000/namanjain12/<repo>_final:<commit>``. This helper auto-prefixes
the registry — pulling the raw Docker-Hub path instead just 504s on a slow miss.

Connection (env vars, with sane defaults for the deployed server):
    OPEN_SANDBOX_DOMAIN   default 10.10.110.50:30080
    OPEN_SANDBOX_REGISTRY default 10.10.110.20:5000   (prefixed onto bare images)
    OPEN_SANDBOX_API_KEY  optional
    OPEN_SANDBOX_PROTOCOL default http

Run as a script to smoke-test against one image (bare or registry-qualified):
    uv run --with opensandbox --python 3.12 env_explore.py \
        namanjain12/aiohttp_final:006fbe03fede4eaa1eeba7b8393cbf4d63cb44b6
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import sys
from datetime import timedelta

from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.sandboxes import SandboxImageAuth, SandboxImageSpec


def _patch_issue_591() -> None:
    """OpenSandbox SDK appends /v1 to the lifecycle base URL → routes through the
    sync path and can 504. Root async lifecycle routes directly under the domain.
    (Same workaround Harbor's opensandbox.py applies.)"""
    if getattr(ConnectionConfig, "_issue_591_patched", False):
        return

    def _base_url(self: ConnectionConfig) -> str:
        domain = self.get_domain()
        if domain.startswith(("http://", "https://")):
            return domain.rstrip("/")
        return f"{self.protocol}://{domain}"

    ConnectionConfig.get_base_url = _base_url
    ConnectionConfig._issue_591_patched = True


_patch_issue_591()


class EnvSandbox:
    """Thin async context manager around an OpenSandbox sandbox for one env image."""

    def __init__(
        self,
        image: str,
        *,
        domain: str | None = None,
        api_key: str | None = None,
        protocol: str | None = None,
        registry_user: str | None = None,
        registry_pass: str | None = None,
        registry: str | None = None,
        cpu: str = "4",
        memory: str = "4Gi",
        sandbox_timeout_min: int = 60,
        ready_timeout_sec: int = 600,
        request_timeout_sec: int = 1800,
    ) -> None:
        reg = registry or os.environ.get("OPEN_SANDBOX_REGISTRY") or "10.10.110.20:5000"
        self.image = _resolve_image(image, reg)
        self._cpu, self._memory = cpu, memory
        self._sandbox_timeout_min = sandbox_timeout_min
        self._ready_timeout_sec = ready_timeout_sec
        self._auth = None
        ru = registry_user or os.environ.get("OPEN_SANDBOX_REGISTRY_USER")
        rp = registry_pass or os.environ.get("OPEN_SANDBOX_REGISTRY_PASS")
        if ru and rp:
            self._auth = SandboxImageAuth(username=ru, password=rp)
        self._conn = ConnectionConfig(
            domain=domain or os.environ.get("OPEN_SANDBOX_DOMAIN") or "10.10.110.50:30080",
            api_key=api_key if api_key is not None else os.environ.get("OPEN_SANDBOX_API_KEY"),
            protocol=protocol or os.environ.get("OPEN_SANDBOX_PROTOCOL") or "http",
            use_server_proxy=True,
            request_timeout=timedelta(seconds=request_timeout_sec),
        )
        self._sandbox: Sandbox | None = None

    async def start(self, attempts: int = 3) -> "EnvSandbox":
        print(f"    base_url={self._conn.get_base_url()}", flush=True)
        last = None
        for i in range(1, attempts + 1):
            try:
                self._sandbox = await Sandbox.create(
                    SandboxImageSpec(image=self.image, auth=self._auth),
                    timeout=timedelta(minutes=self._sandbox_timeout_min),
                    ready_timeout=timedelta(seconds=self._ready_timeout_sec),
                    resource={"cpu": self._cpu, "memory": self._memory},
                    metadata={"managed_by": "rl-env-selection", "image": _k8s_label(self.image)},
                    connection_config=self._conn,
                )
                return self
            except Exception as e:  # 504 on first pull → server keeps pulling; retry
                last = e
                print(f"    [attempt {i}/{attempts}] create failed: {str(e)[:120]}", flush=True)
                if i < attempts:
                    await asyncio.sleep(20)
        raise last

    async def exec(self, command: str, timeout_sec: int = 300) -> tuple[int, str, str]:
        """Run a shell command in the container. Returns (return_code, stdout, stderr)."""
        assert self._sandbox is not None, "call start() first"
        from opensandbox.models.execd import RunCommandOpts

        ex = await self._sandbox.commands.run(
            f"bash -lc {_q(command)}",
            opts=RunCommandOpts(background=False, timeout=timedelta(seconds=timeout_sec)),
        )
        out = "\n".join(m.text for m in ex.logs.stdout) if ex.logs.stdout else ""
        err = "\n".join(m.text for m in ex.logs.stderr) if ex.logs.stderr else ""
        return (1 if ex.error else 0, out, err)

    async def read_file(self, path: str) -> bytes:
        assert self._sandbox is not None, "call start() first"
        return await self._sandbox.files.read_bytes(path)

    async def rollout(
        self,
        task: str,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_turns: int = 30,
        workdir: str = "/testbed",
        install_timeout: int = 600,
        run_timeout: int = 1800,
    ) -> dict:
        """Install + run mini-swe-agent inside the container with instruction `task`.

        `task` is whatever you want the agent to do — typically an EXPLORATION
        instruction ("explore /testbed and report what the issue touches / how hard
        it looks; do not fix it") to gather features, but it can be any instruction.

        In-container path Harbor uses, driven directly via exec — no Harbor
        dataset/verifier needed. The agent's bash actions run natively in `workdir`
        (/testbed = the repo at the buggy commit); model calls go to `model`
        (litellm name, e.g. 'openai/qwen3-32b') at `api_base`.

        Returns: {returncode, stdout, patch (git diff, if any),
                  trajectory (mini-swe-agent JSON str), produced_patch (bool)}.
        Use stdout / trajectory as the agent's findings. Nothing here touches the
        hidden test oracle — you extract predictive features, not a true reward.
        """
        assert self._sandbox is not None, "call start() first"
        # 1. install (idempotent). uv ships its own py3.12 — the container python
        #    (often 3.9) is too old for mini-swe-agent's 3.10+ syntax.
        await self.exec(
            "command -v $HOME/.local/bin/mini-swe-agent >/dev/null 2>&1 || "
            "uv tool install --python 3.12 mini-swe-agent",
            timeout_sec=install_timeout,
        )
        # 2. run the agent
        env = ["export PATH=$PATH:$HOME/.local/bin", "export MSWEA_CONFIGURED=true"]
        if api_base:
            env.append(f"export OPENAI_API_BASE={shlex.quote(api_base)}")
        if api_key:
            env.append(f"export MSWEA_API_KEY={shlex.quote(api_key)} OPENAI_API_KEY={shlex.quote(api_key)}")
        run_cmd = (
            "; ".join(env) + "; "
            + f"cd {shlex.quote(workdir)}; "
            + f"mini-swe-agent --yolo --model {shlex.quote(model)} "
            + f"--task {shlex.quote(task)} --output /tmp/mswea_traj.json "
            + f"--cost-limit 0 --exit-immediately -c agent.max_turns={int(max_turns)} 2>&1"
        )
        rc, out, err = await self.exec(run_cmd, timeout_sec=run_timeout)
        # 3. collect the patch it produced + its trajectory
        _, diff, _ = await self.exec(
            f"cd {shlex.quote(workdir)} && git add -A && git diff --cached", timeout_sec=120
        )
        traj = None
        try:
            traj = (await self.read_file("/tmp/mswea_traj.json")).decode("utf-8", "replace")
        except Exception:
            pass
        return {
            "returncode": rc,
            "stdout": out,
            "patch": diff,
            "trajectory": traj,
            "produced_patch": bool((diff or "").strip()),
        }

    async def stop(self) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.kill()
            finally:
                try:
                    await self._sandbox.close()
                except Exception:
                    pass
                self._sandbox = None

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.stop()


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _resolve_image(image: str, registry: str) -> str:
    """Prefix bare (Docker-Hub-style) images with the internal registry.

    `namanjain12/aiohttp_final:<c>` -> `10.10.110.20:5000/namanjain12/aiohttp_final:<c>`.
    Already-qualified images (first path segment looks like a host: has '.'/':'
    or is 'localhost') are returned unchanged.
    """
    head = image.split("/", 1)[0]
    if "." in head or ":" in head or head == "localhost":
        return image  # already registry-qualified
    return f"{registry}/{image}"


def _k8s_label(value: str) -> str:
    """k8s-label-safe value (<=63 chars, [A-Za-z0-9] ends, only -_. inside)."""
    raw = (value or "").strip()
    san = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("_.-") or "value"
    if len(san) <= 63:
        return san
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{san[:54].rstrip('_.-')}-{digest}"


async def _smoke_test(image: str) -> int:
    print(f"[1/4] creating sandbox from {image} ...", flush=True)
    async with EnvSandbox(image) as env:
        print(f"[2/4] sandbox ready (id={env._sandbox.id})", flush=True)
        for cmd in [
            "echo hello-from-container && whoami && pwd",
            "ls -la /testbed | head -20",
            "cd /testbed && git log --oneline -1 2>/dev/null || echo 'no git'",
            "python -c 'import sys; print(sys.version)'",
        ]:
            rc, out, err = await env.exec(cmd, timeout_sec=120)
            print(f"\n$ {cmd}\n  rc={rc}\n  {out.strip()[:600]}", flush=True)
            if err.strip():
                print(f"  [stderr] {err.strip()[:300]}", flush=True)
        print("\n[3/4] commands done, tearing down ...", flush=True)
    print("[4/4] sandbox killed. OK ✅", flush=True)
    return 0


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else (
        "namanjain12/aiohttp_final:006fbe03fede4eaa1eeba7b8393cbf4d63cb44b6"
    )
    raise SystemExit(asyncio.run(_smoke_test(img)))
