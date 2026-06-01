from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from urllib import error as urlerror
from urllib import parse, request

from .user_runtime import (
    ensure_user_subdir,
    resolve_user_dir,
    user_skills_dir,
    user_tools_dir,
)


URL_SCHEMES = {"http", "https"}


@dataclass(frozen=True, slots=True)
class ManagedResource:
    """A user-managed skill or tool file under ~/.pyagent."""

    path: Path
    label: str


@dataclass(frozen=True, slots=True)
class ResourceInstallResult:
    source: str
    destination: Path
    bytes_written: int


@dataclass(frozen=True, slots=True)
class ResourceKind:
    name: str
    directory_name: str
    default_suffix: str
    allowed_suffixes: tuple[str, ...]
    include_patterns: tuple[str, ...]
    excluded_dirs: tuple[str, ...] = ()


SKILL_KIND = ResourceKind(
    name="skill",
    directory_name="skills",
    default_suffix=".md",
    allowed_suffixes=(".md", ".skill"),
    include_patterns=("*.md", "*.skill"),
)

TOOL_KIND = ResourceKind(
    name="tool",
    directory_name="tools",
    default_suffix=".py",
    allowed_suffixes=(".py",),
    include_patterns=("*.py",),
    excluded_dirs=(".cache", "__pycache__"),
)


def kind_for_name(name: str) -> ResourceKind:
    normalized = name.strip().lower()
    if normalized in {"skill", "skills"}:
        return SKILL_KIND
    if normalized in {"tool", "tools"}:
        return TOOL_KIND
    raise ValueError(f"Unknown resource kind: {name}")


def resource_dir(kind: ResourceKind, user_dir: str | os.PathLike[str] | None = None) -> Path:
    base = resolve_user_dir(user_dir)
    if kind is SKILL_KIND:
        return user_skills_dir(base)
    if kind is TOOL_KIND:
        return user_tools_dir(base)
    return base / kind.directory_name


def _is_url(source: str) -> bool:
    return parse.urlparse(source).scheme.lower() in URL_SCHEMES


def _safe_filename(name: str) -> str:
    candidate = Path(parse.unquote(name)).name.strip()
    candidate = candidate.replace("\x00", "")
    if candidate in {"", ".", ".."}:
        return ""
    return candidate


def _filename_from_url(source: str) -> str:
    parsed = parse.urlparse(source)
    name = _safe_filename(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    if name:
        return name
    host = parsed.netloc.split(":", 1)[0] or "download"
    return _safe_filename(host) or "download"


def _destination_name(source: str, kind: ResourceKind, explicit_name: str | None) -> str:
    name = explicit_name.strip() if explicit_name else ""
    if name:
        candidate = _safe_filename(name)
    elif _is_url(source):
        candidate = _filename_from_url(source)
    else:
        candidate = _safe_filename(Path(source).expanduser().name)

    if not candidate:
        raise ValueError(
            "Could not determine a destination filename. Use --name.")

    suffix = Path(candidate).suffix
    if not suffix:
        candidate = f"{candidate}{kind.default_suffix}"
    return candidate


def _validate_destination_name(name: str, kind: ResourceKind) -> None:
    path = Path(name)
    if path.name != name or name in {"", ".", ".."}:
        raise ValueError(
            "Destination name must be a single filename, not a path. Apparently directories need boundaries too.")
    if path.suffix.lower() not in kind.allowed_suffixes:
        allowed = ", ".join(kind.allowed_suffixes)
        raise ValueError(
            f"{kind.name.title()} files must use one of: {allowed}")


def _read_source(source: str, *, timeout: float = 30.0) -> bytes:
    if _is_url(source):
        req = request.Request(source, headers={"User-Agent": "pyagent"})
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urlerror.URLError as exc:
            raise ValueError(f"Could not download {source}: {exc}") from exc

    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(f"Source file does not exist: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read source file {path}: {exc}") from exc


def list_resources(kind: ResourceKind, *, user_dir: str | os.PathLike[str] | None = None) -> list[ManagedResource]:
    root = resource_dir(kind, user_dir)
    if not root.is_dir():
        return []

    seen: set[Path] = set()
    resources: list[ManagedResource] = []
    excluded = set(kind.excluded_dirs)
    for pattern in kind.include_patterns:
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in excluded for part in relative.parts[:-1]):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            resources.append(ManagedResource(
                path=resolved, label=relative.as_posix()))
    resources.sort(key=lambda item: item.label)
    return resources


def install_resource(
    kind: ResourceKind,
    source: str,
    *,
    user_dir: str | os.PathLike[str] | None = None,
    name: str | None = None,
    force: bool = False,
) -> ResourceInstallResult:
    if not source.strip():
        raise ValueError("Source must not be empty.")

    destination_name = _destination_name(source, kind, name)
    _validate_destination_name(destination_name, kind)
    root = ensure_user_subdir(resource_dir(kind, user_dir))
    destination = root / destination_name

    if destination.exists() and not force:
        raise ValueError(
            f"Destination already exists: {destination}. Use --force to overwrite.")

    data = _read_source(source)
    try:
        tmp = destination.with_name(f".{destination.name}.tmp")
        tmp.write_bytes(data)
        shutil.move(str(tmp), str(destination))
    except OSError as exc:
        raise ValueError(f"Could not write {destination}: {exc}") from exc

    if kind is TOOL_KIND:
        try:
            destination.chmod(destination.stat().st_mode | 0o111)
        except OSError:
            pass

    return ResourceInstallResult(source=source, destination=destination, bytes_written=len(data))


def _candidate_remove_names(kind: ResourceKind, target: str) -> list[str]:
    cleaned = target.strip().strip("/")
    if not cleaned:
        return []
    normalized = Path(cleaned).as_posix()
    names = [normalized]
    if Path(normalized).suffix == "":
        names.append(f"{normalized}{kind.default_suffix}")
    return names


def resolve_resource(kind: ResourceKind, target: str, *, user_dir: str | os.PathLike[str] | None = None) -> ManagedResource | None:
    candidates = set(_candidate_remove_names(kind, target))
    if not candidates:
        return None
    for resource in list_resources(kind, user_dir=user_dir):
        if resource.label in candidates or Path(resource.label).name in candidates:
            return resource
    return None


def remove_resource(kind: ResourceKind, target: str, *, user_dir: str | os.PathLike[str] | None = None) -> Path:
    resource = resolve_resource(kind, target, user_dir=user_dir)
    if resource is None:
        root = resource_dir(kind, user_dir)
        raise ValueError(
            f"No {kind.name} named {target!r} was found under {root}.")
    try:
        resource.path.unlink()
    except OSError as exc:
        raise ValueError(f"Could not remove {resource.path}: {exc}") from exc
    return resource.path
