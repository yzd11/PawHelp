"""Start the local stack in one command: containers, migrations, MCP servers, classifier and app.

Ctrl+C stops every process this script started; containers keep running for a fast restart.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / ".runtime" / "logs"
CONTAINERS = ("mysql", "etcd", "minio", "milvus")
CLASSIFIER_MODEL = ROOT / "data" / "topic-classifier" / "model"
# Local services must never be sent through a system or VPN proxy.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
WINDOWS = os.name == "nt"


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def get_json(url: str) -> dict | None:
    try:
        with OPENER.open(url, timeout=2) as response:
            return json.loads(response.read())
    except (OSError, ValueError):
        return None


def log_tail(path: Path, lines: int = 15) -> str:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return "\n".join(text.splitlines()[-lines:])


class Stack:
    def __init__(self) -> None:
        self.processes: list[tuple[str, subprocess.Popen, Path]] = []
        self.env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    def start(self, name: str, port: int, args: list[str]) -> subprocess.Popen | None:
        if port_open(port):
            print(f"  {name}: 端口 {port} 已有服务在运行，沿用它")
            return None
        LOGS.mkdir(parents=True, exist_ok=True)
        log = LOGS / f"{name}.log"
        # Own process group, so Ctrl+C reaches this script first and children stop in order.
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}
        with log.open("wb") as handle:
            process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=self.env,
                                       stdout=handle, stderr=subprocess.STDOUT, **flags)
        self.processes.append((name, process, log))
        print(f"  {name}: 启动中（日志 {log.relative_to(ROOT)}）")
        return process

    def wait(self, name: str, ready, timeout: float) -> None:
        process = next((p for n, p, _ in self.processes if n == name), None)
        deadline = time.monotonic() + timeout
        while not ready():
            if process is not None and process.poll() is not None:
                self.fail(name)
            if time.monotonic() > deadline:
                raise SystemExit(f"{name} 在 {timeout:.0f}s 内未就绪，见 .runtime/logs/{name}.log")
            time.sleep(0.5)

    def fail(self, name: str) -> None:
        log = next(log for n, _, log in self.processes if n == name)
        print(f"\n{name} 意外退出，日志末尾：\n{log_tail(log)}", file=sys.stderr)
        raise SystemExit(1)

    def stop(self) -> None:
        for name, process, _ in reversed(self.processes):
            if process.poll() is None:
                process.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT)
        for name, process, _ in reversed(self.processes):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            print(f"  {name}: 已停止")


def start_containers() -> None:
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise SystemExit("Docker 未运行：请先启动 Docker Desktop，或加 --no-docker 使用已有服务。")
    print("容器：docker compose up（等待健康检查）")
    subprocess.run(["docker", "compose", "up", "-d", "--wait", *CONTAINERS], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-docker", action="store_true", help="不启动容器，使用已在运行的 MySQL/Milvus")
    parser.add_argument("--no-mcp", action="store_true", help="不启动物流/售后 MCP 服务")
    parser.add_argument("--no-classifier", action="store_true", help="不启动主题分类器（8110）")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    # Ctrl+Break, a closed console or SIGTERM must stop the children too, not orphan them.
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.default_int_handler)
    began = time.monotonic()
    if not args.no_docker:
        start_containers()
    print("数据库迁移：alembic upgrade head")
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, check=True)

    from app.config import get_settings

    settings = get_settings()
    stack = Stack()
    try:
        print("服务：")
        mcp_ports = {}
        if not args.no_mcp and settings.mcp_enabled:
            for name, url in (("logistics", settings.mcp_logistics_url), ("aftersales", settings.mcp_aftersales_url)):
                mcp_ports[name] = urlsplit(url).port
                stack.start(name, mcp_ports[name], ["-m", f"mcp_servers.{name}_server"])
        classifier = not args.no_classifier and (CLASSIFIER_MODEL / "model.safetensors").exists()
        if not args.no_classifier and not classifier:
            print(f"  classifier: 跳过，未找到模型 {CLASSIFIER_MODEL.relative_to(ROOT)}")
        if classifier:
            stack.start("classifier", 8110,
                        ["-m", "uvicorn", "scripts.topics.serve:app", "--host", "127.0.0.1", "--port", "8110"])
        stack.start("app", 8000, ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"])

        for name, port in mcp_ports.items():
            stack.wait(name, lambda port=port: port_open(port), 60)
        if classifier:
            stack.wait("classifier",
                       lambda: (get_json("http://127.0.0.1:8110/healthz") or {}).get("status") == "ok", 120)
        stack.wait("app", lambda: get_json("http://127.0.0.1:8000/health") is not None, 120)
        print(f"页面可用（{time.monotonic() - began:.0f}s）：http://127.0.0.1:8000/  后台 http://127.0.0.1:8000/admin")
        print("加载检索模型（BGE-M3 + 重排）……")
        stack.wait("app", lambda: (get_json("http://127.0.0.1:8000/health") or {}).get("knowledge_models", "ready")
                   != "loading", 600)
        models = (get_json("http://127.0.0.1:8000/health") or {}).get("knowledge_models", "ready")
        if models == "failed":
            print("检索模型预热失败，首次提问时会再尝试加载；原因见 .runtime/logs/app.log", file=sys.stderr)
        print(f"全部就绪（{time.monotonic() - began:.0f}s）。按 Ctrl+C 停止本脚本启动的服务；容器继续运行。")
        while True:
            for name, process, _ in stack.processes:
                if process.poll() is not None:
                    stack.fail(name)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止……")
    finally:
        stack.stop()


if __name__ == "__main__":
    main()
