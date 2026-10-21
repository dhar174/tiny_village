"""Regenerate the Python 3.12 Linux x86-64 requirements using pinned uv."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("core", "visual", "dev", "analytics", "nlp", "retrieval", "llm-cpu", "llm-gpu")
UV_VERSION = "0.12.23"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*", choices=PROFILES,
                        help="Default: core, visual and dev; core is always compiled first.")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Run with Python 3.12 to preserve the supported lock target.")
    try:
        installed_uv = version("uv")
    except PackageNotFoundError:
        parser.error("Install the locked development profile to obtain the uv resolver.")
    if installed_uv != UV_VERSION:
        parser.error(f"Use uv=={UV_VERSION} from the locked development profile.")
    selected = args.profiles or ["core", "visual", "dev"]
    profiles = ["core", *dict.fromkeys(p for p in selected if p != "core")]
    output_dir = ROOT / "requirements" / "locks"
    output_dir.mkdir(exist_ok=True)
    for profile in profiles:
        command = [sys.executable, "-m", "uv", "pip", "compile",
                   f"requirements/{profile}.in", "--python-version", "3.12",
                   "--python-platform", "x86_64-manylinux_2_28",
                   "--generate-hashes", "--only-binary", ":all:",
                   "--index-url", "https://pypi.org/simple",
                   "--output-file", f"requirements/locks/{profile}-py312-linux-x86_64.txt",
                   "--custom-compile-command",
                   f"python scripts/compile_requirements.py {profile}"]
        if profile != "core":
            command += ["--constraint", "requirements/locks/core-py312-linux-x86_64.txt"]
        if profile.startswith("llm-"):
            backend = "cpu" if profile == "llm-cpu" else "cu126"
            # uv binds only PyTorch packages to this index during resolution.
            # A global resolver index would shadow newer PyPI dependencies.
            command += ["--torch-backend", backend, "--emit-index-url"]
        subprocess.run(command, cwd=ROOT, check=True)
        if profile.startswith("llm-"):
            # uv does not emit the scoped backend index in requirements format.
            # Preserve every resolved version/hash and expose its wheel source
            # to ordinary pip, which installs this fully hash-locked closure.
            lock = output_dir / f"{profile}-py312-linux-x86_64.txt"
            content = lock.read_text()
            index = f"--extra-index-url https://download.pytorch.org/whl/{backend}"
            if index not in content:
                content = content.replace("--index-url https://pypi.org/simple\n",
                                          "--index-url https://pypi.org/simple\n" + index + "\n", 1)
                lock.write_text(content)


if __name__ == "__main__":
    main()
