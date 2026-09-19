import tarfile
from pathlib import Path

import pytest

from pier.agents.installed.codex import Codex
from pier.cli.ssh import _rewrite_paths_and_build_payload
from pier.environments.base import ExecResult
from pier.models.agent.context import AgentContext
from pier.models.job.config import JobConfig
from pier.models.trial.config import AgentConfig, TaskConfig


def test_codex_force_auth_json_resolves_the_local_subscription_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codex_home = tmp_path / "home" / ".codex"
    codex_home.mkdir(parents=True)
    auth_path = codex_home / "auth.json"
    auth_path.write_text('{"tokens": "subscription"}')

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.setenv("CODEX_FORCE_AUTH_JSON", "1")

    agent = Codex(logs_dir=tmp_path / "logs")

    assert agent.resolve_auth_json_path() == auth_path


def test_codex_auth_path_fails_before_a_remote_run_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.setenv("CODEX_FORCE_AUTH_JSON", "true")

    agent = Codex(logs_dir=tmp_path / "logs")

    with pytest.raises(ValueError, match="CODEX_FORCE_AUTH_JSON is set"):
        agent.resolve_auth_json_path()


@pytest.mark.asyncio
async def test_codex_subscription_mode_uploads_auth_json_instead_of_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codex_home = tmp_path / "home" / ".codex"
    codex_home.mkdir(parents=True)
    auth_path = codex_home / "auth.json"
    auth_path.write_text('{"tokens": "subscription"}')
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class FakeEnvironment:
        default_user = None

        def __init__(self) -> None:
            self.exec_calls: list[dict[str, object]] = []
            self.uploads: list[tuple[Path | str, str]] = []

        def agent_process_env(self, env):
            return env

        async def exec(self, **kwargs):
            self.exec_calls.append(kwargs)
            return ExecResult(return_code=0, stdout="", stderr="")

        async def upload_file(self, source_path, target_path):
            self.uploads.append((source_path, target_path))

    environment = FakeEnvironment()
    agent = Codex(
        logs_dir=tmp_path / "logs",
        model_name="gpt-5.5",
        extra_env={"CODEX_FORCE_AUTH_JSON": "1"},
    )

    await agent.run("Fix the task", environment, AgentContext())

    assert environment.uploads == [(auth_path, "/tmp/codex-secrets/auth.json")]
    assert all("OPENAI_API_KEY" not in (call.get("env") or {}) for call in environment.exec_calls)


def test_ssh_payload_transfers_codex_auth_without_serializing_its_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do the task")
    (task_dir / "task.toml").write_text("[environment]\n")

    codex_home = tmp_path / "home" / ".codex"
    codex_home.mkdir(parents=True)
    auth_path = codex_home / "auth.json"
    auth_path.write_text('{"access_token": "subscription-secret"}')
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.setenv("CODEX_FORCE_AUTH_JSON", "1")

    config = JobConfig(
        job_name="auth-test",
        tasks=[TaskConfig(path=task_dir)],
        agents=[
            AgentConfig(
                name="codex",
                env={"CODEX_FORCE_AUTH_JSON": "1"},
            )
        ],
    )
    task_configs = [TaskConfig(path=task_dir)]
    payload_path = tmp_path / "payload.tar"

    remote_config = _rewrite_paths_and_build_payload(
        config,
        task_configs,
        remote_root="/tmp/pier-ssh-test",
        payload_path=payload_path,
    )

    auth_env = remote_config.agents[0].env["CODEX_AUTH_JSON_PATH"]
    assert auth_env == "${PIER_SSH_CODEX_AUTH_0}"

    with tarfile.open(payload_path) as archive:
        names = archive.getnames()
        config_text = archive.extractfile("config.json").read().decode()
        env_text = archive.extractfile("env.sh").read().decode()

    assert "codex-auth/0/auth.json" in names
    assert "subscription-secret" not in config_text
    assert "subscription-secret" not in env_text
    assert str(auth_path) not in config_text
