"""Download only the frozen cohort using unsigned public S3 HTTPS requests."""

import csv
import http.client
import json
import os
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/nih_effusion_vs_nofinding_v1.csv"
OUTPUT = ROOT / "data/processed/nih/images_224x224"
REPORT = ROOT / "data/processed/nih/download_verification_v1.json"
EXPECTED = 19326
WORKERS = 8
S3_HOST = "s3.us-east-2.amazonaws.com"
S3_PREFIX = "/nih-chest-x-rays/images_224x224/"
THREAD_STATE = threading.local()


def download(name):
    destination = OUTPUT / name
    if destination.is_file() and destination.stat().st_size > 0:
        return "skipped", name, None
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            # No credentials or Authorization header: unsigned public S3 access.
            if not hasattr(THREAD_STATE, "connection"):
                THREAD_STATE.connection = http.client.HTTPSConnection(
                    S3_HOST, timeout=60
                )
            connection = THREAD_STATE.connection
            connection.request("GET", S3_PREFIX + quote(name, safe=""))
            with connection.getresponse() as response:
                if response.status != 200:
                    raise OSError(f"S3 HTTP {response.status}: {response.reason}")
                expected_size = response.getheader("Content-Length")
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            size = temporary.stat().st_size
            if size == 0:
                raise OSError("Downloaded an empty file")
            if expected_size is not None and size != int(expected_size):
                raise OSError(f"Incomplete download: {size}/{expected_size} bytes")
            os.replace(temporary, destination)
            return "downloaded", name, None
        except Exception as error:
            if hasattr(THREAD_STATE, "connection"):
                THREAD_STATE.connection.close()
                del THREAD_STATE.connection
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                return "failed", name, str(error)
            time.sleep(attempt + 1)


def main():
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert "Image Index" in reader.fieldnames, "Missing Image Index column"
        names = [row["Image Index"] for row in reader]
    assert len(names) == EXPECTED, f"Expected {EXPECTED} rows, found {len(names)}"
    assert len(names) == len(set(names)), "Duplicate Image Index values"
    assert all(name and Path(name).name == name and name not in {".", ".."}
               and "/" not in name and "\\" not in name for name in names), (
        "Invalid image filename"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures = []
    print(f"Requested: {len(names)} images; unsigned S3; workers: {WORKERS}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(download, name) for name in names]
        for completed, future in enumerate(as_completed(futures), 1):
            status, name, error = future.result()
            counts[status] += 1
            if error:
                failures.append({"filename": name, "error": error})
                print(f"FAILED: {name}: {error}", flush=True)
            if completed % 500 == 0 or completed == EXPECTED:
                print(f"Progress: {completed}/{EXPECTED}; downloaded={counts['downloaded']}; "
                      f"skipped={counts['skipped']}; failed={counts['failed']}", flush=True)

    local_files = [path for path in OUTPUT.iterdir() if path.is_file()]
    local_names = [path.name for path in local_files]
    missing = sorted(name for name in names if not (OUTPUT / name).is_file())
    zero_bytes = sorted(path.name for path in local_files if path.stat().st_size == 0)
    duplicates = len(local_names) - len(set(local_names))
    extras = sorted(set(local_names) - set(names))
    report = {
        "expected_images": EXPECTED,
        "files_present": len(local_files),
        "downloaded_this_run": counts["downloaded"],
        "skipped_existing": counts["skipped"],
        "failed_downloads": counts["failed"],
        "missing_images": missing,
        "zero_byte_files": zero_bytes,
        "duplicate_filenames": duplicates,
        "unexpected_files": extras,
        "download_errors": sorted(failures, key=lambda item: item["filename"]),
        "output_directory": OUTPUT.relative_to(ROOT).as_posix(),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verification report:", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    for label, values in (("MISSING", missing), ("ZERO-BYTE", zero_bytes), ("UNEXPECTED", extras)):
        for name in values:
            print(f"{label}: {name}", flush=True)
    if missing or zero_bytes or duplicates or extras or len(local_files) != EXPECTED or failures:
        print("FAIL: download verification did not pass.", flush=True)
        return 1
    print("PASS: all 19326 frozen-cohort images are present and nonzero.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
