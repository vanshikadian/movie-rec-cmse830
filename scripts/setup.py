"""Create a local environment and prepare missing MovieLens runtime files."""

import hashlib
import shutil
import ssl
import subprocess
import sys
import urllib.request
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
ARCHIVE_MD5 = "d472be332d4daa821edc399621853b57"
RUNTIME = [
    f"data/processed/{name}.parquet" for name in ("ratings", "movies", "movie_genres", "split")
]
RUNTIME += ["models/matrix_factorization.npz", "models/metrics.json"]


def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Use Python 3.11 for this project.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    ready = all((ROOT / name).is_file() for name in RUNTIME)
    requirements = ROOT / ("requirements-demo.txt" if ready else "requirements.txt")
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
    if not ready:
        raw = ROOT / "data/raw/ml-32m"
        files = ("ratings.csv", "movies.csv", "tags.csv", "links.csv")
        if not all((raw / name).is_file() for name in files):
            raw.mkdir(parents=True, exist_ok=True)
            archive = raw.parent / "ml-32m.zip"
            print("Downloading MovieLens 32M from GroupLens (about 239 MB).", flush=True)
            print(
                "Research use only; see docs/source/ml-32m_README.txt for the conditions.",
                flush=True,
            )
            if not archive.exists():
                partial = archive.with_suffix(".zip.part")
                certificates = subprocess.check_output(
                    [str(python), "-m", "certifi"], text=True
                ).strip()
                context = ssl.create_default_context(cafile=certificates)
                with (
                    urllib.request.urlopen(ARCHIVE_URL, context=context, timeout=120) as source,
                    partial.open("wb") as dest,
                ):
                    shutil.copyfileobj(source, dest)
                partial.replace(archive)
            digest = hashlib.md5()
            with archive.open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != ARCHIVE_MD5:
                raise SystemExit(
                    "MovieLens archive checksum mismatch. Move the archive aside and rerun setup."
                )
            with zipfile.ZipFile(archive) as source:
                for name in (*files, "README.txt", "checksums.txt"):
                    with source.open(f"ml-32m/{name}") as src, (raw / name).open("wb") as dest:
                        shutil.copyfileobj(src, dest)
        subprocess.run([str(python), str(ROOT / "scripts/prepare_data.py")], cwd=ROOT, check=True)
        subprocess.run([str(python), str(ROOT / "scripts/train_models.py")], cwd=ROOT, check=True)
    print("\nSetup complete. Start the app with:")
    print(f'"{python}" -m streamlit run "{ROOT / "app.py"}"')


if __name__ == "__main__":
    main()
