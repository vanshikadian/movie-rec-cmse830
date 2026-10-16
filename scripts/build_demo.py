"""Build an excluded local demo archive with saved runtime files and dataset terms."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "app.py",
    "README.md",
    "images/both-pick.png",
    "requirements-demo.txt",
    ".streamlit/config.toml",
    "assets/both-pick.css",
    "scripts/setup.py",
    "scripts/check_app.py",
    "scripts/check_group.py",
    "docs/source/ml-32m_README.txt",
    "docs/source/ml-32m_checksums.txt",
    "models/matrix_factorization.npz",
    "models/metrics.json",
]
FILES += [str(f.relative_to(ROOT)) for f in sorted((ROOT / "src").glob("*.py"))]
FILES += [
    f"data/processed/{name}.parquet" for name in ("ratings", "movies", "movie_genres", "split")
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/both-pick-demo.zip")
    output = parser.parse_args().output
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing demo files: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as tmp:
        temporary = Path(tmp.name)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in FILES:
            bundle.write(ROOT / name, "both-pick-demo/" + name)
        bundle.writestr("both-pick-demo/manifest.json", json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output)
    print(f"Demo: {output} ({output.stat().st_size / 1_000_000:.1f} MB)")
    print(
        "Includes transformed MovieLens data and learned model parameters. Keep the bundled GroupLens conditions."
    )


if __name__ == "__main__":
    main()
