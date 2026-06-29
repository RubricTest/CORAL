import asyncio
from env_explore import EnvSandbox
IMAGE = "namanjain12/aiohttp_final:006fbe03fede4eaa1eeba7b8393cbf4d63cb44b6"
async def main():
    async with EnvSandbox(IMAGE) as env:
        for cmd in [
            "uv tool install --python 3.12 --force mini-swe-agent 2>&1 | tail -4",
            "export PATH=$PATH:$HOME/.local/bin; mini-swe-agent --help 2>&1 | head -35",
        ]:
            rc,out,err = await env.exec(cmd, timeout_sec=300)
            print(f"### {cmd[:55]}  rc={rc}\n{(out or '')[:1800]}\n[stderr]{(err or '')[:400]}\n", flush=True)
asyncio.run(main())
