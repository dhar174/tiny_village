#!/usr/bin/env python3
"""Rebuild local review artifacts/archive only; never approve or apply GitHub changes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def retained_files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.name not in ("artifact-hashes.json", "portable-package.zip")
                  and not p.name.startswith(("live-plan", "local-checkpoint"))
                  and p.suffix not in (".lock", ".tmp", ".zip"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", required=True,
                        help="Generate a new review bundle/digest; this is not human approval")
    parser.parse_args()
    subprocess.run([sys.executable, "-B", str(ROOT / "build_preview.py")], check=True)
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in retained_files()}
    (ROOT / "artifact-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    subprocess.run([sys.executable, "-B", str(ROOT / "validate_package.py")], check=True)
    archive = ROOT / "portable-package.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for p in retained_files() + [ROOT / "artifact-hashes.json"]:
            member = zipfile.ZipInfo("tiny-village-project-737-publication/" + p.relative_to(ROOT).as_posix().replace("\\", "/"), date_time=(2026, 10, 23, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            output.writestr(member, p.read_bytes())
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "github_mutations_performed": 0,
                      "approval": "not supplied by rebuilding"}))


if __name__ == "__main__":
    main()
