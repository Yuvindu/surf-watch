import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("status") != "complete":
        errors.append(f"manifest status is {manifest.get('status')!r}, not 'complete'")

    checkpoints = manifest.get("checkpoints")
    if not checkpoints:
        errors.append("manifest does not contain checkpoint hashes")
        return errors

    checkpoint_dir = manifest_path.parent / "checkpoints"
    for record in checkpoints:
        filename = Path(record["path"]).name
        local_path = checkpoint_dir / filename
        if not local_path.is_file():
            errors.append(f"missing checkpoint: {local_path}")
            continue
        if local_path.stat().st_size != record["bytes"]:
            errors.append(f"size mismatch: {local_path}")
            continue
        if sha256_file(local_path) != record["sha256"]:
            errors.append(f"SHA-256 mismatch: {local_path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify copied training checkpoints against their manifest"
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    errors = verify_manifest(args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Verified all checkpoint artifacts in {args.manifest.parent}")


if __name__ == "__main__":
    main()
