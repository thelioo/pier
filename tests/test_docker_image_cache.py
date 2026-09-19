import asyncio
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

from pier.environments.docker import docker as docker_module
from pier.environments.docker.docker import DockerEnvironment, _build_context_fingerprint


def test_context_fingerprint_is_independent_of_context_path(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "Dockerfile").write_text("FROM alpine\nCOPY . /app\n")
        (root / "nested" / "run.sh").write_text("#!/bin/sh\necho ok\n")

    assert _build_context_fingerprint(first) == _build_context_fingerprint(second)

    (second / "nested" / "run.sh").write_text("#!/bin/sh\necho changed\n")
    assert _build_context_fingerprint(first) != _build_context_fingerprint(second)


def test_context_fingerprint_changes_when_file_mode_changes(tmp_path: Path):
    context = tmp_path / "context"
    context.mkdir()
    script = context / "run.sh"
    script.write_text("#!/bin/sh\necho ok\n")

    before = _build_context_fingerprint(context)
    script.chmod(0o755)

    assert before != _build_context_fingerprint(context)


def test_cached_image_name_has_content_suffix_and_bounded_length():
    name = DockerEnvironment._cached_image_name("a" * 300, "0123456789abcdef")

    assert name.endswith("__ctx-0123456789abcdef")
    assert len(name) <= 255


def test_image_exists_checks_the_docker_daemon(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)

    assert DockerEnvironment._image_exists("pier-cache:latest") is True
    assert calls == [["docker", "image", "inspect", "pier-cache:latest"]]


def test_image_exists_treats_missing_docker_image_as_a_cache_miss(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["docker", "image", "inspect", "missing"],
            returncode=1,
        )

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)

    assert DockerEnvironment._image_exists("missing") is False


def test_force_build_always_bypasses_image_cache():
    assert DockerEnvironment._should_build(
        force_build=True,
        cache_enabled=True,
        image_exists=True,
    ) is True
    assert DockerEnvironment._should_build(
        force_build=False,
        cache_enabled=True,
        image_exists=True,
    ) is False
    assert DockerEnvironment._should_build(
        force_build=False,
        cache_enabled=True,
        image_exists=False,
    ) is True
    assert DockerEnvironment._should_build(
        force_build=False,
        cache_enabled=False,
        image_exists=True,
    ) is True


def test_start_skips_compose_build_when_cached_image_exists(monkeypatch):
    environment = object.__new__(DockerEnvironment)
    commands: list[list[str]] = []
    environment._prepare_agent_build_context = lambda: None
    environment._prepare_egress_proxy_compose = lambda: None
    environment._write_resources_compose_file = lambda: None
    environment._mounts_json = []
    environment.agent_install_spec = None
    environment._use_prebuilt = False
    environment._set_cached_image_name = lambda: True
    environment._validate_daemon_mode = lambda: None
    environment._image_exists = lambda _image_name: True
    environment._validate_image_os = lambda _image_name: asyncio.sleep(0)
    environment._run_docker_compose_command = lambda command: _record_command(commands, command)
    environment._env_vars = SimpleNamespace(main_image_name="pier-cache")
    environment._image_cache_enabled = False
    environment._is_windows_container = True
    environment.environment_name = "same-task"
    environment.task_env_config = SimpleNamespace(docker_image=None)
    environment.logger = logging.getLogger("test-docker-image-cache")

    asyncio.run(environment.start(force_build=False))

    assert [command[0] for command in commands] == ["down", "up"]


def _record_command(commands: list[list[str]], command: list[str]):
    commands.append(command)
    return asyncio.sleep(0)
