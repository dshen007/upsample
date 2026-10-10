"""Verify the extracted ROS2 PNG dataset without opening the source bags."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = {"source": str(root.resolve()), "bags": [], "total_frames": 0,
              "total_png_bytes": 0, "all_checks_passed": False}
    for bag in manifest["bags"]:
        name = Path(bag["source_bag"]).name
        with (root / name / "frames.csv").open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        if len(records) != bag["exported_frames"]:
            raise AssertionError(f"Frame count mismatch: {name}")
        by_sequence = defaultdict(list)
        for record in records:
            relative = Path(record["relative_path"])
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise AssertionError(f"Frame path missing/outside output: {relative}")
            encoded = path.read_bytes()
            if digest(encoded) != record["png_sha256"]:
                raise AssertionError(f"PNG hash mismatch: {path}")
            pixels = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
            if pixels is None or pixels.shape != (512, 640) or pixels.dtype != np.uint8:
                raise AssertionError(f"PNG format mismatch: {path}")
            if digest(pixels.tobytes()) != record["pixel_sha256"]:
                raise AssertionError(f"Pixel hash mismatch: {path}")
            by_sequence[record["sequence"]].append(record)
            result["total_png_bytes"] += len(encoded)
        sequence_checks = []
        for sequence in bag["sequences"]:
            rows = by_sequence[sequence["name"]]
            if len(rows) != sequence["frames"]:
                raise AssertionError(f"Sequence count mismatch: {sequence['name']}")
            stamps = [int(r["bag_timestamp_ns"]) for r in rows]
            if any(b <= a for a, b in zip(stamps, stamps[1:])):
                raise AssertionError(f"Non-increasing frame timestamps: {sequence['name']}")
            max_gap = max(((b-a)/1e6 for a, b in zip(stamps, stamps[1:])), default=0)
            if "/clip_" in sequence["name"] and max_gap > 100:
                raise AssertionError(f"Dense clip has >100ms gap: {sequence['name']}")
            sequence_checks.append({"sequence": sequence["name"], "frames": len(rows),
                                    "max_gap_ms": round(max_gap, 3),
                                    "unique_pixel_frames": len({r["pixel_sha256"] for r in rows})})
        result["bags"].append({"bag": name, "frames": len(records),
                               "sequences": sequence_checks})
        result["total_frames"] += len(records)
    if result["total_frames"] != 1404:
        raise AssertionError("Unexpected frozen v1 total frame count")
    result["all_checks_passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    (args.root / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": result["total_frames"],
                      "png_mb": round(result["total_png_bytes"]/1e6, 1),
                      "bags": [(b["bag"], b["frames"]) for b in result["bags"]],
                      "passed": result["all_checks_passed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
