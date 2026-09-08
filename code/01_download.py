"""Download the four PhysioNet databases used in this study.

Databases
---------
mitdb   MIT-BIH Arrhythmia Database                      (development, DS1/DS2)
incartdb St Petersburg INCART 12-lead Arrhythmia Database (external validation)
svdb    MIT-BIH Supraventricular Arrhythmia Database      (external validation)
nstdb   MIT-BIH Noise Stress Test Database                (noise sweep)

Nothing is redistributed with the code; this script only fetches.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import wfdb

DATA_ROOT = Path(os.environ.get("ECG_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))

DBS = ["mitdb", "incartdb", "svdb", "nstdb"]


def already_have(dst: Path) -> int:
    return len(list(dst.glob("*.hea")))


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for db in DBS:
        dst = DATA_ROOT / db
        dst.mkdir(parents=True, exist_ok=True)
        n = already_have(dst)
        if n > 0:
            print(f"[skip] {db}: {n} headers already present", flush=True)
            continue
        print(f"[get ] {db} -> {dst}", flush=True)
        t0 = time.time()
        for attempt in range(1, 4):
            try:
                wfdb.dl_database(db, str(dst))
                break
            except Exception as exc:  # noqa: BLE001
                print(f"       attempt {attempt} failed: {exc}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(5)
        print(f"[done] {db}: {already_have(dst)} headers in {time.time() - t0:.0f}s", flush=True)
    print("ALL_DOWNLOADS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
