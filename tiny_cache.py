"""Prepare user-owned Hugging Face caches without importing model libraries."""

import os
from pathlib import Path
import stat


def _cache_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value == "":
        raise ValueError(f"{name} is empty")
    path = Path(value if value is not None else default).expanduser()
    if ".." in path.parts:
        raise ValueError(f"parent traversal ('..') is not supported in {name}; use the direct directory path")
    return path.absolute()


def _validate_directory_path(path: Path) -> None:
    # Check all existing components before changing any selected directory.
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise ValueError(f"symlink or reparse directory is not supported: {component}")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"not a directory: {component}")
        if component == path and os.name == "posix" and info.st_uid != os.geteuid():
            raise ValueError(f"directory is not owned by the runtime user: {path}")
        if component != path and os.name == "posix":
            _validate_ancestor_owner(info, component)


def _validate_ancestor_owner(info: os.stat_result, path: Path) -> None:
    # An ancestor's owner can replace entries even when its mode is 0755.
    if info.st_uid not in (0, os.geteuid()):
        raise ValueError(f"cache ancestor is not owned by the runtime user or root: {path}")


def _prepare_directory(path: Path) -> None:
    if os.name != "posix":
        # Windows directory access is controlled by inherited ACLs, not chmod.
        _validate_directory_path(path)
        for component in (*reversed(path.parents), path):
            _validate_directory_path(component)
            component.mkdir(mode=0o700, exist_ok=True)
            _validate_directory_path(component)
        return

    # Walk using descriptors so a substituted symlink cannot redirect chmod.
    directory_flags = os.O_DIRECTORY | os.O_NOFOLLOW
    search_flags = getattr(os, "O_SEARCH", getattr(os, "O_PATH", os.O_RDONLY)) | directory_flags
    descriptor = os.open(path.anchor, search_flags)
    try:
        for index, part in enumerate(path.parts[1:], start=1):
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            # Traversing an ancestor requires search permission, not listing.
            flags = (os.O_RDONLY | directory_flags) if index == len(path.parts) - 1 else search_flags
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            info = os.fstat(descriptor)
            if index < len(path.parts) - 1:
                _validate_ancestor_owner(info, Path(path.anchor, *path.parts[1:index + 1]))
                if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
                    raise ValueError(f"cache parent allows other users to replace entries: {path}")
                continue
            if info.st_uid != os.geteuid():
                raise ValueError(f"directory is not owned by the runtime user: {path}")
            if stat.S_IMODE(info.st_mode) != 0o700:
                os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


def configure_model_cache() -> str:
    """Return a prepared model-cache path, preserving explicit HF settings.

    Call before importing Hugging Face libraries, which snapshot environment
    settings. Existing owned cache directories become private on POSIX; their
    files and parent permissions are never modified. No shared cache is migrated.
    """
    path = "configured model cache"
    try:
        if "HF_HOME" in os.environ:
            default_home = Path(os.environ["HF_HOME"])
        else:
            default_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
        home = _cache_path("HF_HOME", default_home)
        hub = _cache_path("HF_HUB_CACHE", home / "hub")
        model = _cache_path("TRANSFORMERS_CACHE", hub)
        paths = list(dict.fromkeys((home, hub, model)))
        for path in paths:
            if path == Path(path.anchor):
                raise ValueError("a filesystem root cannot be a private model cache")
            _validate_directory_path(path)
        for path in paths:
            _prepare_directory(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot prepare owned model cache at {path}: {error}. "
            "Set HF_HOME, HF_HUB_CACHE or TRANSFORMERS_CACHE to a directory owned "
            "by the runtime user beneath a trusted parent."
        ) from error
    # Publish defaults only after all selected paths have passed validation.
    os.environ.setdefault("HF_HOME", str(home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    return str(model)
