#!/usr/bin/env python3
"""Download and safely extract official ERIM raw category archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
BASE_URL = (
    "https://www.chicagobooth.edu/boothsitecore/docs/acnpanel/"
    "raw_files/erim/{category}/{category}.zip"
)
CATEGORIES = (
    "brownie", "ddinner", "ketchup", "marg", "pbutter", "sugar", "tissue", "tuna")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        with zipfile.ZipFile(destination) as archive:
            bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"existing archive {destination} has a corrupt member: {bad}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "energy-basket-research/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"downloaded archive has a corrupt member: {bad}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_extract(archive_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(
            prefix="erim_extract_", dir=destination.parent) as temporary:
        temporary_root = Path(temporary).resolve()
        members = [member for member in archive.infolist() if not member.is_dir()]
        for member in members:
            candidate = (temporary_root / member.filename).resolve()
            if temporary_root not in candidate.parents:
                raise ValueError(f"unsafe archive member: {member.filename!r}")
            if Path(member.filename).name != member.filename:
                raise ValueError(f"nested archive member is not supported: {member.filename!r}")
        archive.extractall(temporary_root)
        for member in members:
            source = temporary_root / member.filename
            target = destination / member.filename
            if target.exists():
                if target.is_file() and sha256(target) == sha256(source):
                    extracted.append(target)
                    continue
                raise FileExistsError(
                    f"refusing to replace a different existing ERIM file: {target}")
            os.replace(source, target)
            extracted.append(target)
    return sorted(extracted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", choices=CATEGORIES,
                        help="category to fetch; repeat as needed (default: all)")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/erim_basket")
    args = parser.parse_args()
    categories = tuple(dict.fromkeys(args.category or CATEGORIES))
    output = args.output_dir.resolve()
    manifest = {"source": "Chicago Booth Kilts Center ERIM archive", "categories": {}}
    for category in categories:
        url = BASE_URL.format(category=category)
        archive_path = output / "archives" / f"{category}.zip"
        print(f"[erim-fetch] {category}: {url}", flush=True)
        download(url, archive_path)
        files = safe_extract(archive_path, output / "raw" / category)
        manifest["categories"][category] = {
            "url": url,
            "archive": str(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256(archive_path),
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in files
            },
        }
        print(f"[erim-fetch] {category}: {len(files)} files", flush=True)
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[erim-fetch] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
