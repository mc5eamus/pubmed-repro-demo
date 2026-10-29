"""Download and checksum the public Framingham and UCI Cleveland source files."""

import hashlib
import json
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SOURCES = {
    "framingham.csv": {
        "url": "https://raw.githubusercontent.com/plaguedoc000/framingham-heart-study-dataset/main/framingham.csv",
        "sha256": "2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018",
    },
    "processed.cleveland.data": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "sha256": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    },
}


def download_data(raw_dir: Path = RAW_DIR) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename, source in SOURCES.items():
        request = urllib.request.Request(
            source["url"], headers={"User-Agent": "raman2026-chd-reproduction/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != source["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {source['sha256']}, got {checksum}"
            )
        (raw_dir / filename).write_bytes(content)
        manifest[filename] = {"url": source["url"], "sha256": checksum}
    (raw_dir / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    for name, info in download_data().items():
        print(f"{name}: {info['url']} (sha256: {info['sha256']})")
