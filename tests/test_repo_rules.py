"""Guards for rules that disqualify a submission if broken."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_api_publishes_port_8080():
    assert "8080:8080" in COMPOSE["services"]["api"]["ports"]


def test_only_the_api_publishes_a_port():
    # A published Postgres port could clash on a clean machine and exposes the database.
    # Workers must publish nothing, or they couldn't be scaled past one copy.
    assert "ports" not in COMPOSE["services"]["db"]
    assert "ports" not in COMPOSE["services"]["worker"]


def test_worker_can_be_scaled():
    # A fixed name would stop `--scale worker=N` from working.
    assert "container_name" not in COMPOSE["services"]["worker"]
    assert COMPOSE["services"]["worker"]["deploy"]["replicas"] >= 2
    assert COMPOSE["services"]["worker"]["depends_on"]["db"]["condition"] == "service_healthy"


def test_api_waits_for_a_healthy_database():
    assert COMPOSE["services"]["api"]["depends_on"]["db"]["condition"] == "service_healthy"
    assert "healthcheck" in COMPOSE["services"]["db"]


@pytest.mark.parametrize("service", ["api", "worker"])
def test_llm_settings_are_passed_through_without_defaults(service):
    env = COMPOSE["services"][service]["environment"]
    assert env["LLM_BASE_URL"] == "${LLM_BASE_URL:-}"
    assert env["LLM_API_KEY"] == "${LLM_API_KEY:-}"


def test_compose_does_not_require_an_env_file():
    # A required env_file would break `docker compose up` on a clean machine.
    assert "env_file" not in COMPOSE["services"]["api"]
    assert "env_file" not in COMPOSE["services"]["worker"]


@pytest.mark.parametrize("path", ["app", "Dockerfile", "docker-compose.yml"])
def test_no_hardcoded_keys_or_provider_endpoints(path):
    target = ROOT / path
    files = [target] if target.is_file() else target.rglob("*.py")
    pattern = re.compile(r"sk-or-|sk-[A-Za-z0-9]{16,}|openrouter\.ai|api\.openai\.com|api\.anthropic\.com")
    for f in files:
        assert not pattern.search(f.read_text()), f"{f} contains a key or provider endpoint"


@pytest.mark.parametrize("path", [".env", "sample/manifest.csv", "sample/schemas/boxscore.json"])
def test_secrets_and_sample_data_are_git_ignored(path):
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("no git repository")
    assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


def test_docker_context_excludes_secrets_and_sample_data():
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    assert "*" in lines, "the build context should be an allowlist"
    assert not any(line.startswith("!") and ("sample" in line or ".env" in line) for line in lines)
