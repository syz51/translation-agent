from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from uuid import uuid4

import psycopg
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_docker_available() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is unavailable; skipping Postgres integration tests")

    checks = (
        (["docker", "compose", "version"], "docker compose is unavailable; skipping Postgres integration tests"),
        (["docker", "info"], "docker daemon is unavailable; skipping Postgres integration tests"),
    )
    for command, message in checks:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            if detail:
                message = f"{message}: {detail}"
            pytest.skip(message)


def _wait_for_postgres(dsn: str, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return
        except Exception as exc:  # pragma: no cover - exercised in polling loop
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"postgres did not become ready in time: {last_error}")


@pytest.fixture
def postgres_dsn() -> str:
    _ensure_docker_available()

    compose_file = ROOT / "compose.yaml"
    project_name = f"translation-agent-{uuid4().hex[:8]}"
    port = _find_free_port()
    user = "translation_agent"
    password = "translation_agent"
    dbname = "translation_agent_test"
    env = {
        **os.environ,
        "TA_PG_PORT": str(port),
        "TA_PG_USER": user,
        "TA_PG_PASSWORD": password,
        "TA_PG_DB": dbname,
    }
    compose_command = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "-p",
        project_name,
    ]

    try:
        subprocess.run(
            [*compose_command, "up", "-d", "postgres"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        dsn = f"postgresql://{user}:{password}@127.0.0.1:{port}/{dbname}?connect_timeout=1"
        _wait_for_postgres(dsn)
        yield dsn
    finally:
        subprocess.run(
            [*compose_command, "down", "-v", "--remove-orphans"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
