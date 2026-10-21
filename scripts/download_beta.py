"""Download script for BETA dataset from Figshare (DOI: 10.6084/m9.figshare.12264401).

Downloads S1.mat through S10.mat (or more) into data/raw/, along with
documentation files (Description.pdf, note.txt).
"""

import os
import sys
import argparse
import urllib.request
import json
from pathlib import Path

FIGSHARE_ARTICLE_URL = "https://api.figshare.com/v2/articles/12264401"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Fallback direct URLs in case Figshare API is unreachable or rate-limited
KNOWN_FILES = {
    "Description.pdf": "https://ndownloader.figshare.com/files/35903012",
    "note.txt": "https://ndownloader.figshare.com/files/22618172",
    "S1.mat": "https://ndownloader.figshare.com/files/22620932",
    "S2.mat": "https://ndownloader.figshare.com/files/22620938",
    "S3.mat": "https://ndownloader.figshare.com/files/22620944",
    "S4.mat": "https://ndownloader.figshare.com/files/22620947",
    "S5.mat": "https://ndownloader.figshare.com/files/22620950",
    "S6.mat": "https://ndownloader.figshare.com/files/22620956",
    "S7.mat": "https://ndownloader.figshare.com/files/22621247",
    "S8.mat": "https://ndownloader.figshare.com/files/22618196",
    "S9.mat": "https://ndownloader.figshare.com/files/22620971",
    "S10.mat": "https://ndownloader.figshare.com/files/22620983",
    "S11.mat": "https://ndownloader.figshare.com/files/22620986",
    "S12.mat": "https://ndownloader.figshare.com/files/22620989",
    "S13.mat": "https://ndownloader.figshare.com/files/22620992",
    "S14.mat": "https://ndownloader.figshare.com/files/22620995",
    "S15.mat": "https://ndownloader.figshare.com/files/22620998",
}


def get_figshare_file_list():
    """Query Figshare API for all files in the BETA article."""
    req = urllib.request.Request(
        FIGSHARE_ARTICLE_URL,
        headers={"User-Agent": "OPT098-BETA-Downloader/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {f["name"]: (f["download_url"], f["size"]) for f in data.get("files", [])}
    except Exception as e:
        print(f"[WARN] Failed to fetch file list from Figshare API ({e}). Using predefined URLs.")
        return {name: (url, 0) for name, url in KNOWN_FILES.items()}


def download_file(url: str, dest_path: Path, expected_size: int = 0):
    """Download a file with download progress reporting."""
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if expected_size > 0 and actual_size == expected_size:
            print(f"[SKIP] {dest_path.name} already downloaded ({actual_size / (1024*1024):.1f} MB).")
            return True
        elif actual_size > 50 * 1024 * 1024:  # roughly 50MB+ for .mat
            print(f"[SKIP] {dest_path.name} already exists ({actual_size / (1024*1024):.1f} MB).")
            return True

    print(f"[DOWNLOADING] {dest_path.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "OPT098-BETA-Downloader/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", expected_size or 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            last_percent = -1

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = int(downloaded * 100 / total_size)
                    if percent % 20 == 0 and percent != last_percent:
                        print(f"  -> {dest_path.name}: {percent}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)")
                        last_percent = percent

        print(f"[DONE] Successfully downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB).")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # remove partially downloaded file
        return False


def main():
    parser = argparse.ArgumentParser(description="Download BETA SSVEP dataset samples.")
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=10,
        help="Number of subjects to download (default: 10, recommended 10-15)"
    )
    parser.add_argument(
        "--start-subject",
        type=int,
        default=1,
        help="Starting subject index (default: 1)"
    )
    parser.add_argument(
        "--all-metadata",
        action="store_true",
        default=True,
        help="Download documentation files (Description.pdf, note.txt)"
    )
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {RAW_DATA_DIR}")

    file_map = get_figshare_file_list()

    # 1. Download metadata/documentation
    if args.all_metadata:
        for meta_file in ["Description.pdf", "note.txt"]:
            if meta_file in file_map:
                url, size = file_map[meta_file]
                download_file(url, RAW_DATA_DIR / meta_file, size)
            elif meta_file in KNOWN_FILES:
                download_file(KNOWN_FILES[meta_file], RAW_DATA_DIR / meta_file)

    # 2. Download subjects
    success_count = 0
    target_count = args.num_subjects

    for i in range(args.start_subject, args.start_subject + target_count):
        subject_name = f"S{i}.mat"
        if subject_name in file_map:
            url, size = file_map[subject_name]
        elif subject_name in KNOWN_FILES:
            url, size = KNOWN_FILES[subject_name], 0
        else:
            print(f"[WARN] No URL found for {subject_name}")
            continue

        if download_file(url, RAW_DATA_DIR / subject_name, size):
            success_count += 1

    print(f"\nFinished: {success_count}/{target_count} subjects downloaded successfully into {RAW_DATA_DIR}.")


if __name__ == "__main__":
    main()
