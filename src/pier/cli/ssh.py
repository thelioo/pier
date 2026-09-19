from __future__ import annotations

import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from rich.console import Console
from typer import Argument, Typer

from pier.models.job.config import JobConfig
from pier.models.job.result import JobResult
from pier.models.trial.config import TaskConfig
from pier.utils.env import get_required_host_vars, is_env_template

ssh_app = Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

_REMOTE_PIER = '"$HOME/.cache/pier/venv/bin/pier"'
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ssh(
    target: str,
    command: str,
    *,
    tty: bool = False,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    args = ["ssh"]
    if tty:
        args.append("-tt")
    args.extend([target, command])
    return subprocess.run(
        args,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def _sftp(target: str, commands: list[str]) -> subprocess.CompletedProcess[str]:
    batch = "\n".join(commands) + "\n"
    return subprocess.run(
        ["sftp", "-q", "-b", "-", target],
        input=batch,
        check=False,
        text=True,
        capture_output=True,
    )


def _raise_ssh_error(result: subprocess.CompletedProcess[str], action: str) -> None:
    if result.returncode == 0:
        return
    details = (result.stderr or result.stdout or "").strip()
    suffix = f": {details}" if details else ""
    raise RuntimeError(f"SSH {action} failed with exit code {result.returncode}{suffix}")


def _sftp_path(path: Path | str) -> str:
    return shlex.quote(str(path))


def _remote_root() -> str:
    return f"/tmp/pier-ssh-{uuid4().hex}"


def _project_root() -> Path:
    current = Path.cwd().resolve()
    source = Path(__file__).resolve()
    candidates = (*[current, *current.parents], *source.parents)
    for candidate in candidates:
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists() and 'name = "datacurve-pier"' in pyproject.read_text():
            return candidate
    raise RuntimeError(
        "Could not find the Pier checkout. Run this command from the Pier repository "
        "or install the local wheel before using `pier ssh setup`."
    )


def _build_wheel(output_dir: Path) -> Path:
    project_root = _project_root()
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(output_dir)],
            cwd=project_root,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`uv` is required to build the local Pier wheel for SSH setup."
        ) from exc

    wheels = sorted(output_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("`uv build` completed without producing a wheel.")
    return wheels[-1]


def _remote_setup_command(remote_upload: str) -> str:
    remote_upload_q = shlex.quote(remote_upload)
    return " ".join(
        [
            "set -eu;",
            'mkdir -p "$HOME/.cache/pier";',
            'if ! python3 -m venv "$HOME/.cache/pier/venv"; then',
            '  if command -v apt-get >/dev/null 2>&1; then',
            '    if [ "$(id -u)" -eq 0 ]; then',
            '      apt-get update && apt-get install -y python3-venv;',
            '    elif command -v sudo >/dev/null 2>&1; then',
            '      sudo -n apt-get update && sudo -n apt-get install -y python3-venv;',
            '    else',
            '      echo "python3-venv is required and passwordless sudo is unavailable" >&2; exit 1;',
            '    fi;',
            '    rm -rf -- "$HOME/.cache/pier/venv";',
            '    python3 -m venv "$HOME/.cache/pier/venv";',
            '  else',
            '    echo "python3-venv is required to create the Pier worker environment" >&2; exit 1;',
            '  fi;',
            'fi;',
            '"$HOME/.cache/pier/venv/bin/python" -m pip install '
            '--disable-pip-version-check --upgrade',
            remote_upload_q,
            ";",
            '"$HOME/.cache/pier/venv/bin/pier" --version;',
            'if command -v docker >/dev/null 2>&1; then docker --version; fi;',
            f"rm -rf -- {shlex.quote(str(Path(remote_upload).parent))};",
        ]
    )


def setup_remote(target: str) -> None:
    """Build this checkout and install the resulting wheel on *target*."""
    remote_root = _remote_root()
    with tempfile.TemporaryDirectory(prefix="pier-ssh-setup-") as temp_dir:
        wheel = _build_wheel(Path(temp_dir))
        # Keep the wheel's canonical filename.  pip validates the filename
        # before installing it, so a generic name such as ``pier.whl`` is not
        # accepted as a wheel distribution.
        remote_wheel = f"{remote_root}/{wheel.name}"

        try:
            result = _ssh(
                target,
                f"mkdir -p -- {shlex.quote(remote_root)} && chmod 700 -- {shlex.quote(remote_root)}",
                capture_output=True,
            )
            _raise_ssh_error(result, "setup directory creation")

            result = _sftp(
                target,
                [f"put {_sftp_path(wheel)} {_sftp_path(remote_wheel)}"],
            )
            _raise_ssh_error(result, "wheel upload")

            result = _ssh(target, _remote_setup_command(remote_wheel))
            _raise_ssh_error(result, "remote Pier installation")
        finally:
            _ssh(target, f"rm -rf -- {shlex.quote(remote_root)}", capture_output=True)


def _safe_env_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value).upper()
    return result or "VALUE"


def _add_template_values(
    mapping: dict[str, str],
    env_values: dict[str, str],
) -> None:
    """Make host values referenced by an env mapping available on the VPS."""
    for value in mapping.values():
        if not is_env_template(value):
            continue
        for name, default in get_required_host_vars({"value": value}):
            if name in os.environ:
                env_values[name] = os.environ[name]
            elif default is None:
                raise ValueError(
                    f"Environment variable '{name}' is required by the remote run "
                    "but is not present on the local machine."
                )


def _externalize_env_mapping(
    mapping: dict[str, str],
    env_values: dict[str, str],
    *,
    scope: str,
    skip_keys: set[str] | None = None,
) -> None:
    """Move literal job-level env values into the temporary remote env script."""
    skip_keys = skip_keys or set()
    for index, (key, value) in enumerate(list(mapping.items())):
        if key in skip_keys:
            continue
        if is_env_template(value):
            _add_template_values({key: value}, env_values)
            continue

        remote_name = f"PIER_SSH_{scope.upper()}_{index}_{_safe_env_name(key)}"
        env_values[remote_name] = value
        mapping[key] = f"${{{remote_name}}}"


def _collect_task_env_values(
    task_configs: list[TaskConfig],
    env_values: dict[str, str],
) -> None:
    from pier.models.task.task import Task

    for task_config in task_configs:
        task = Task(task_config.get_local_path())
        dumped = task.config.model_dump(mode="python")
        _walk_env_mappings(dumped, env_values)


def _walk_env_mappings(value: Any, env_values: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "env" and isinstance(child, dict):
                _add_template_values(child, env_values)
            _walk_env_mappings(child, env_values)
    elif isinstance(value, list):
        for child in value:
            _walk_env_mappings(child, env_values)


def _collect_provider_env_values(
    config: JobConfig,
    env_values: dict[str, str],
) -> None:
    from pier.agents.utils import get_api_key_var_names_from_model_name

    for agent in config.agents:
        if not agent.model_name:
            continue
        try:
            provider_keys = get_api_key_var_names_from_model_name(agent.model_name)
        except ValueError:
            continue
        for key in provider_keys:
            if key in os.environ:
                env_values[key] = os.environ[key]

    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "GEMINI_API_BASE",
        "GOOGLE_GEMINI_BASE_URL",
    ):
        if key in os.environ:
            env_values[key] = os.environ[key]


def _resolve_local_codex_auth(
    config: JobConfig,
    index: int,
    logs_dir: Path,
) -> Path | None:
    from pier.agents.factory import AgentFactory
    from pier.agents.installed.codex import Codex
    from pier.models.agent.name import AgentName

    agent_config = config.agents[index]
    if agent_config.name != AgentName.CODEX.value:
        return None

    agent = AgentFactory.create_agent_from_config(
        agent_config,
        logs_dir=logs_dir / f"agent-{index}",
    )
    if not isinstance(agent, Codex):
        return None
    return agent.resolve_auth_json_path()


def _stage_local_path(
    source: Path,
    *,
    remote_root: str,
    archive_name: str,
    staged: dict[Path, str],
    archive_entries: list[tuple[Path, str]],
) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Local path required by the remote run does not exist: {source}")

    if source in staged:
        return Path(remote_root) / staged[source]

    staged[source] = archive_name
    archive_entries.append((source, archive_name))
    return Path(remote_root) / archive_name


def _rewrite_paths_and_build_payload(
    config: JobConfig,
    task_configs: list[TaskConfig],
    *,
    remote_root: str,
    payload_path: Path,
) -> JobConfig:
    remote_config = config.model_copy(deep=True)
    remote_config.jobs_dir = Path(remote_root) / "jobs"

    archive_entries: list[tuple[Path, str]] = []
    staged: dict[Path, str] = {}
    for index, task_config in enumerate(remote_config.tasks):
        if task_config.path is None:
            raise ValueError("SSH execution only supports local task paths.")
        task_config.path = _stage_local_path(
            task_config.path,
            remote_root=remote_root,
            archive_name=f"tasks/{index}",
            staged=staged,
            archive_entries=archive_entries,
        )

    for index, dataset in enumerate(remote_config.datasets):
        if dataset.path is None:
            raise ValueError("SSH execution only supports local dataset paths.")
        dataset.path = _stage_local_path(
            dataset.path,
            remote_root=remote_root,
            archive_name=f"datasets/{index}",
            staged=staged,
            archive_entries=archive_entries,
        )

    if remote_config.environment.mounts:
        for index, mount in enumerate(remote_config.environment.mounts):
            if mount.get("type") != "bind":
                continue
            source = Path(mount["source"]).expanduser()
            if source.exists():
                mount["source"] = str(
                    _stage_local_path(
                        source,
                        remote_root=remote_root,
                        archive_name=f"mounts/{index}",
                        staged=staged,
                        archive_entries=archive_entries,
                    )
                )

    env_values: dict[str, str] = {}
    _externalize_env_mapping(
        remote_config.environment.env,
        env_values,
        scope="environment",
    )
    _externalize_env_mapping(
        remote_config.verifier.env,
        env_values,
        scope="verifier",
    )
    _collect_task_env_values(task_configs, env_values)
    _collect_provider_env_values(config, env_values)

    auth_files: list[tuple[int, Path, str]] = []
    with tempfile.TemporaryDirectory(prefix="pier-ssh-auth-") as auth_logs:
        for index, agent in enumerate(remote_config.agents):
            auth_path = _resolve_local_codex_auth(
                config,
                index,
                Path(auth_logs),
            )
            _externalize_env_mapping(
                agent.env,
                env_values,
                scope=f"agent_{index}",
                skip_keys=(
                    {"CODEX_AUTH_JSON_PATH"}
                    if agent.name == "codex"
                    else None
                ),
            )
            if auth_path is None:
                continue

            remote_auth = f"{remote_root}/codex-auth/{index}/auth.json"
            remote_env_name = f"PIER_SSH_CODEX_AUTH_{index}"
            agent.env["CODEX_AUTH_JSON_PATH"] = f"${{{remote_env_name}}}"
            env_values[remote_env_name] = remote_auth
            auth_files.append((index, auth_path.resolve(), f"codex-auth/{index}/auth.json"))

        env_script = payload_path.parent / "env.sh"
        env_script.write_text(
            "#!/bin/sh\n"
            + "set -eu\n"
            + "".join(
                _format_export(key, value) for key, value in sorted(env_values.items())
            )
        )
        env_script.chmod(0o600)

        config_path = payload_path.parent / "config.json"
        config_path.write_text(remote_config.model_dump_json(indent=4))
        config_path.chmod(0o600)

        with tarfile.open(payload_path, mode="w") as archive:
            archive.add(config_path, arcname="config.json", recursive=False)
            archive.add(env_script, arcname="env.sh", recursive=False)
            for source, archive_name in archive_entries:
                archive.add(source, arcname=archive_name, recursive=True)
            for _index, auth_path, archive_name in auth_files:
                archive.add(auth_path, arcname=archive_name, recursive=False)

    return remote_config


def _format_export(key: str, value: str) -> str:
    if not _ENV_NAME_RE.fullmatch(key):
        raise ValueError(f"Invalid environment variable name for SSH transfer: {key!r}")
    return f"export {key}={shlex.quote(value)}\n"


def _remote_extract_command(remote_root: str) -> str:
    root = shlex.quote(remote_root)
    return (
        f"set -eu; tar --extract --file {root}/payload.tar --directory {root} "
        f"--no-same-owner; rm -f -- {root}/payload.tar; chmod 600 {root}/env.sh; "
        f"find {root}/codex-auth -type f -name auth.json -exec chmod 600 {{}} + 2>/dev/null || true"
    )


def _remote_run_command(remote_root: str, *, yes: bool) -> str:
    root = shlex.quote(remote_root)
    yes_arg = " --yes" if yes else ""
    return (
        "set -u; "
        f"root={root}; "
        "trap 'rm -f -- \"$root/env.sh\" "
        "\"$root\"/codex-auth/*/auth.json 2>/dev/null || true' EXIT; "
        " . \"$root/env.sh\"; "
        "set +e; "
        f"{_REMOTE_PIER} run --config \"$root/config.json\"{yes_arg}; "
        "status=$?; "
        "exit $status"
    )


def _download_job(
    target: str,
    remote_job_dir: str,
    local_job_dir: Path,
) -> None:
    local_job_dir.parent.mkdir(parents=True, exist_ok=True)
    result = _sftp(
        target,
        [
            f"get -r {_sftp_path(remote_job_dir)} {_sftp_path(local_job_dir.parent)}",
        ],
    )
    _raise_ssh_error(result, "job result download")


def run_remote_job(
    config: JobConfig,
    task_configs: list[TaskConfig],
    *,
    target: str,
    yes: bool,
) -> JobResult:
    """Run the existing Pier job on an SSH worker and download its results."""
    local_job_dir = (config.jobs_dir / config.job_name).expanduser().resolve()
    if local_job_dir.exists():
        raise FileExistsError(
            f"Job directory {local_job_dir} already exists; choose a different --job-name."
        )

    remote_root = _remote_root()
    with tempfile.TemporaryDirectory(prefix="pier-ssh-run-") as temp_dir:
        payload_path = Path(temp_dir) / "payload.tar"
        _rewrite_paths_and_build_payload(
            config,
            task_configs,
            remote_root=remote_root,
            payload_path=payload_path,
        )

        result = _ssh(
            target,
            f"mkdir -p -- {shlex.quote(remote_root)} && chmod 700 -- {shlex.quote(remote_root)} "
            f"&& test -x \"$HOME/.cache/pier/venv/bin/pier\"",
            capture_output=True,
        )
        _raise_ssh_error(
            result,
            "worker check (run `pier ssh setup TARGET` first)",
        )

        try:
            result = _sftp(
                target,
                [f"put {_sftp_path(payload_path)} {_sftp_path(remote_root + '/payload.tar')}"],
            )
            _raise_ssh_error(result, "job payload upload")

            result = _ssh(target, _remote_extract_command(remote_root), capture_output=True)
            _raise_ssh_error(result, "job payload extraction")

            result = _ssh(
                target,
                _remote_run_command(remote_root, yes=True),
                tty=True,
            )
            run_returncode = result.returncode

            remote_job_dir = f"{remote_root}/jobs/{config.job_name}"
            _download_job(target, remote_job_dir, local_job_dir)
        finally:
            _ssh(target, f"rm -rf -- {shlex.quote(remote_root)}", capture_output=True)

    result_path = local_job_dir / "result.json"
    if not result_path.exists():
        if run_returncode != 0:
            raise RuntimeError(
                f"Remote Pier exited with code {run_returncode} and did not produce result.json."
            )
        raise RuntimeError(f"Remote job did not produce {result_path}.")

    # Keep the local job metadata usable by the normal viewer and CLI commands.
    (local_job_dir / "config.json").write_text(config.model_dump_json(indent=4))
    if run_returncode != 0:
        raise RuntimeError(f"Remote Pier exited with code {run_returncode}.")
    return JobResult.model_validate_json(result_path.read_text())


@ssh_app.command("setup")
def setup_command(
    target: str = Argument(help="SSH target accepted by the system ssh command."),
) -> None:
    """Install the current Pier wheel in the SSH worker's private virtualenv."""
    setup_remote(target)
    console.print(f"[green]Pier is ready on {target}.[/green]")
