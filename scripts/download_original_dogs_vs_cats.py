from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:  # pragma: no cover - handled at runtime
    KaggleApi = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
COMPETITION_SLUG = "dogs-vs-cats"
COMPETITION_URL = "https://www.kaggle.com/competitions/dogs-vs-cats"
OUTPUT_DIR = ROOT / "data" / "dogs_vs_cats_original"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the original Kaggle Dogs vs Cats competition data for the NumPy lab."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to place the extracted competition files.",
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


def extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def count_labeled_images(train_dir: Path) -> tuple[int, int]:
    cat_count = 0
    dog_count = 0
    for path in train_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.name.startswith("cat."):
            cat_count += 1
        elif path.name.startswith("dog."):
            dog_count += 1
    return cat_count, dog_count


def main() -> None:
    args = parse_args()
    require_kaggle()

    output_dir: Path = args.output_dir
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"{output_dir} already exists. Re-run with --force to replace it.")
        shutil.rmtree(output_dir)

    api = authenticate()
    with tempfile.TemporaryDirectory(prefix="dogs-vs-cats-download-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        print(f"Downloading competition files for {COMPETITION_SLUG} ...")
        try:
            api.competition_download_files(COMPETITION_SLUG, path=str(tmp_dir), quiet=False)
        except Exception as exc:  # pragma: no cover - depends on Kaggle competition access
            raise SystemExit(
                "Kaggle could not download the competition files. Make sure you have accepted "
                f"the competition rules first: {COMPETITION_URL}"
            ) from exc

        outer_archives = sorted(tmp_dir.glob("*.zip"))
        if not outer_archives:
            raise SystemExit(
                "Download finished, but no zip archives were found. "
                f"Check the competition page: {COMPETITION_URL}"
            )

        competition_extract_dir = tmp_dir / "competition_extract"
        competition_extract_dir.mkdir(parents=True, exist_ok=True)
        for archive in outer_archives:
            extract_zip(archive, competition_extract_dir)

        train_archive_candidates = sorted(competition_extract_dir.rglob("train.zip"))
        if not train_archive_candidates:
            raise SystemExit(
                "The competition files were downloaded, but train.zip was not found after extraction. "
                f"Check the competition page: {COMPETITION_URL}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(train_archive_candidates[0], output_dir)

    train_dir = output_dir / "train"
    if not train_dir.exists():
        raise SystemExit("Extraction finished, but the expected train/ folder was not created.")

    cat_count, dog_count = count_labeled_images(train_dir)
    if cat_count == 0 or dog_count == 0:
        raise SystemExit("The extracted train/ folder does not contain both cat and dog images.")

    print(f"Original Dogs vs Cats data is ready at {output_dir}")
    print(f"{'train/cat.*':>14}: {cat_count}")
    print(f"{'train/dog.*':>14}: {dog_count}")
    print(f"{'source':>14}: {COMPETITION_URL}")
    print("Note: Kaggle competition downloads require you to accept the competition rules first.")


if __name__ == "__main__":
    main()
