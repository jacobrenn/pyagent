from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from urllib import error as urlerror
from urllib import parse, request
import uuid
from zipfile import BadZipFile, ZipFile

from .user_runtime import ensure_user_subdir, resolve_user_dir, user_extensions_dir


MAX_EXTENSION_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_EXTENSION_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_EXTENSION_ARCHIVE_FILES = 1_000
_EXTENSION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_ALLOWED_UPLOAD_SUFFIXES = {".py", ".zip"}


@dataclass(frozen=True, slots=True)
class ExtensionInstallResult:
    name: str
    destination: Path
    bytes_written: int


def normalize_extension_name(value: str) -> str:
    """Validate a single extension name suitable for an import and directory."""
    name = value.strip()
    if name.endswith(".py"):
        name = name[:-3]
    if not _EXTENSION_NAME.fullmatch(name):
        raise ValueError(
            "Extension name must start with a letter or underscore and contain only "
            "letters, digits, underscores, or hyphens."
        )
    return name


def install_extension_bytes(
    data: bytes,
    *,
    source_name: str,
    name: str | None = None,
    force: bool = False,
    user_dir: str | os.PathLike[str] | None = None,
) -> ExtensionInstallResult:
    """Install a single-file extension or a zipped extension package."""
    if len(data) > MAX_EXTENSION_UPLOAD_BYTES:
        raise ValueError(
            f"Extension upload exceeds the {MAX_EXTENSION_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    filename = Path(parse.unquote(source_name)).name.strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError(
            "Extension uploads must be a .py file or .zip archive.")

    inferred = Path(filename).stem
    extension_name = normalize_extension_name(name or inferred)
    root = ensure_user_subdir(
        user_extensions_dir(resolve_user_dir(user_dir))
    )

    active_file = root / f"{extension_name}.py"
    active_package = root / extension_name
    disabled_file = root / "disabled" / f"{extension_name}.py"
    disabled_package = root / "disabled" / extension_name
    conflicts = [
        path
        for path in (active_file, active_package, disabled_file, disabled_package)
        if path.exists()
    ]
    if conflicts and not force:
        raise ValueError(
            f"Extension `{extension_name}` already exists at {conflicts[0]}. "
            "Use force to replace it."
        )

    if suffix == ".py":
        destination = active_file
        tmp = root / f".{extension_name}.{uuid.uuid4().hex}.tmp"
        try:
            tmp.write_bytes(data)
            _remove_paths(conflicts)
            os.replace(tmp, destination)
        except (OSError, ValueError) as exc:
            tmp.unlink(missing_ok=True)
            if isinstance(exc, ValueError):
                raise
            raise ValueError(
                f"Could not install extension at {destination}: {exc}") from exc
        return ExtensionInstallResult(
            name=extension_name,
            destination=destination.resolve(),
            bytes_written=len(data),
        )

    destination = active_package
    tmp_package = root / f".{extension_name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp_package.mkdir()
        bytes_written = _extract_extension_archive(
            data,
            tmp_package,
            preferred_name=extension_name,
        )
        _remove_paths(conflicts)
        tmp_package.rename(destination)
    except (OSError, ValueError) as exc:
        shutil.rmtree(tmp_package, ignore_errors=True)
        if isinstance(exc, ValueError):
            raise
        raise ValueError(
            f"Could not install extension at {destination}: {exc}") from exc

    return ExtensionInstallResult(
        name=extension_name,
        destination=destination.resolve(),
        bytes_written=bytes_written,
    )


def install_extension_url(
    source_url: str,
    *,
    name: str | None = None,
    force: bool = False,
    user_dir: str | os.PathLike[str] | None = None,
    timeout: float = 30.0,
) -> ExtensionInstallResult:
    """Download and install a direct .py or .zip extension URL."""
    parsed = parse.urlparse(source_url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Extension URLs must use HTTP or HTTPS.")
    filename = Path(parse.unquote(parsed.path)).name
    if Path(filename).suffix.lower() not in _ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError(
            "Direct extension URLs must point to a .py file or .zip archive. "
            "Use the repository installer for Git repository URLs."
        )

    req = request.Request(source_url, headers={"User-Agent": "pyagent"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = response.read(MAX_EXTENSION_UPLOAD_BYTES + 1)
    except urlerror.URLError as exc:
        raise ValueError(f"Could not download {source_url}: {exc}") from exc

    return install_extension_bytes(
        data,
        source_name=filename,
        name=name,
        force=force,
        user_dir=user_dir,
    )


def _extract_extension_archive(
    data: bytes,
    destination: Path,
    *,
    preferred_name: str,
) -> int:
    try:
        archive = ZipFile(BytesIO(data))
    except BadZipFile as exc:
        raise ValueError(
            f"Extension upload is not a valid zip archive: {exc}") from exc

    with archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        if not files:
            raise ValueError("Extension zip archive is empty.")
        if len(files) > MAX_EXTENSION_ARCHIVE_FILES:
            raise ValueError(
                f"Extension zip archive contains more than {MAX_EXTENSION_ARCHIVE_FILES} files."
            )

        safe_entries: list[tuple[object, PurePosixPath]] = []
        declared_bytes = 0
        for info in files:
            path = _safe_archive_path(info.filename)
            mode = info.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise ValueError(
                    f"Extension zip archive contains a symbolic link: {path}")
            declared_bytes += info.file_size
            if declared_bytes > MAX_EXTENSION_ARCHIVE_BYTES:
                raise ValueError(
                    f"Expanded extension archive exceeds the "
                    f"{MAX_EXTENSION_ARCHIVE_BYTES // (1024 * 1024)} MB limit."
                )
            safe_entries.append((info, path))

        package_root = _select_package_root(
            [path for _, path in safe_entries], preferred_name=preferred_name
        )
        written = 0
        extracted_files = 0
        for info, path in safe_entries:
            try:
                relative = path.relative_to(package_root)
            except ValueError:
                continue
            if not relative.parts:
                continue
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                while True:
                    chunk = source.read(64 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_EXTENSION_ARCHIVE_BYTES:
                        raise ValueError(
                            f"Expanded extension archive exceeds the "
                            f"{MAX_EXTENSION_ARCHIVE_BYTES // (1024 * 1024)} MB limit."
                        )
                    output.write(chunk)
            extracted_files += 1

        if extracted_files == 0 or not (destination / "__init__.py").is_file():
            raise ValueError(
                "Extension zip archive must contain a package with an __init__.py file."
            )
        return written


def _safe_archive_path(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"Unsafe path in extension zip archive: {value!r}")
    if path.parts[0] == "__MACOSX":
        # It is harmless, but keeping it out avoids treating metadata as package data.
        return PurePosixPath(".__pyagent_metadata__", *path.parts[1:])
    return path


def _select_package_root(
    paths: list[PurePosixPath], *, preferred_name: str
) -> PurePosixPath:
    roots = [path.parent for path in paths if path.name == "__init__.py"]
    if not roots:
        raise ValueError(
            "Extension zip archive must contain a package with an __init__.py file."
        )

    preferred = [root for root in roots if root.name == preferred_name]
    candidates = preferred or roots
    minimum_depth = min(len(root.parts) for root in candidates)
    shallowest = sorted(
        {root for root in candidates if len(root.parts) == minimum_depth},
        key=str,
    )
    if len(shallowest) != 1:
        choices = ", ".join(
            str(root) or "<archive root>" for root in shallowest)
        raise ValueError(
            "Extension zip archive contains multiple possible package roots: "
            f"{choices}. Upload one extension package at a time."
        )
    return shallowest[0]


def _remove_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError as exc:
            raise ValueError(
                f"Could not replace existing extension {path}: {exc}") from exc
