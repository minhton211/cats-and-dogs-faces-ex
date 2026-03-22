from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:  # pragma: no cover - handled at runtime
    KaggleApi = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "duongtranhai/cats-dogs-faces-small"
DATASET_URL = "https://www.kaggle.com/datasets/duongtranhai/cats-dogs-faces-small/data"
OUTPUT_DIR = ROOT / "data" / "cats_dogs_faces_small"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the curated Kaggle cats-dogs-faces-small dataset for the lab notebooks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to place the extracted dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def require_kaggle() -> None:
    if KaggleApi is None:
        raise SystemExit("Missing required package: kaggle. Install dependencies first, for example with `uv sync`.")


def authenticate() -> KaggleApi:
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local auth state
        raise SystemExit(
            "Kaggle authentication failed. Make sure ~/.kaggle/kaggle.json exists "
            "or that KAGGLE_USERNAME and KAGGLE_KEY are set."
        ) from exc
    return api


def image_count(folder: Path) -> int:
    return sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def find_dataset_root(extract_dir: Path) -> Path:
    candidates = []
    for candidate in [extract_dir, *[path for path in extract_dir.iterdir() if path.is_dir()]]:
        expected = [
            candidate / "train" / "cat",
            candidate / "train" / "dog",
            candidate / "val" / "cat",
            candidate / "val" / "dog",
            candidate / "test" / "cat",
            candidate / "test" / "dog",
            candidate / "metadata.csv",
        ]
        if all(path.exists() for path in expected):
            candidates.append(candidate)

    if not candidates:
        raise SystemExit(
            "Download finished, but the extracted files do not match the expected dataset structure. "
            f"Check the Kaggle dataset page: {DATASET_URL}"
        )

    return candidates[0]


def validate_dataset_dir(dataset_dir: Path) -> dict[str, int]:
    counts = {}
    total = 0
    for split in ("train", "val", "test"):
        for label in ("cat", "dog"):
            folder = dataset_dir / split / label
            count = image_count(folder)
            counts[f"{split}/{label}"] = count
            total += count

    if total == 0:
        raise SystemExit("The downloaded dataset contains zero images. Please re-run the download.")

    return counts


def main() -> None:
    args = parse_args()
    require_kaggle()

    output_dir: Path = args.output_dir
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"{output_dir} already exists. Re-run with --force to replace it.")
        shutil.rmtree(output_dir)

    api = authenticate()
    with tempfile.TemporaryDirectory(prefix="cats-dogs-faces-download-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        print(f"Downloading {DATASET_SLUG} ...")
        api.dataset_download_files(DATASET_SLUG, path=str(tmp_dir), unzip=True, quiet=False)

        extracted_root = find_dataset_root(tmp_dir)
        shutil.move(str(extracted_root), str(output_dir))

    counts = validate_dataset_dir(output_dir)
    print(f"Dataset is ready at {output_dir}")
    for key in sorted(counts):
        print(f"{key:>10}: {counts[key]}")
    print(f"{'metadata.csv':>10}: present")
    print(f"{'source':>10}: {DATASET_URL}")


if __name__ == "__main__":
    main()
