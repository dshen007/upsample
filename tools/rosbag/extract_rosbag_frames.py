"""Extract a compact, reproducible infrared frame dataset from ROS2 bags.

Run this script under WSL with rosbags, numpy, OpenCV and PyYAML installed.
The source bags are read only. Output PNGs contain exact 8-bit Y values from
the YUV422 image messages; no color conversion, contrast stretch or resize.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from inspect_flight_rosbags import image_array, topic_info


PLAN = {
    "infrared_20260916_111549": {"overview_interval_s": 2, "dense_fractions": (0.1, 0.5, 0.9)},
    "infrared_20260916_112100": {"overview_interval_s": 2, "dense_fractions": (0.1, 0.5, 0.9)},
    "infrared_bag": {"overview_interval_s": 5, "dense_fractions": (0.5,)},
}
DENSE_DURATION_NS = 6_000_000_000
FIELDS = ("sequence", "frame_index", "bag_timestamp_ns", "header_timestamp_ns",
          "offset_s", "relative_path", "png_sha256", "pixel_sha256", "width",
          "height", "encoding")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_frame(root, sequence, index, bag_start, timestamp, message):
    pixels = image_array(message)
    if pixels.shape != (512, 640) or pixels.dtype != np.uint8:
        raise ValueError(f"Unexpected infrared image format: {pixels.shape}, {pixels.dtype}")
    if message.encoding.lower() != "yuv422":
        raise ValueError(f"Expected audited UYVY encoding, got {message.encoding}")
    relative = Path(sequence) / f"frame_{index:05d}.png"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or not cv2.imwrite(str(target), pixels,
                                          [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Cannot write new PNG: {target}")
    header = message.header.stamp
    header_ns = int(header.sec)*1_000_000_000 + int(header.nanosec)
    return {"sequence": sequence, "frame_index": index,
            "bag_timestamp_ns": int(timestamp), "header_timestamp_ns": header_ns,
            "offset_s": round((timestamp-bag_start)/1e9, 6),
            "relative_path": relative.as_posix(), "png_sha256": sha256_file(target),
            "pixel_sha256": sha256_bytes(pixels.tobytes()), "width": 640,
            "height": 512, "encoding": message.encoding}


def describe_sequence(sequence, records, requested_start_s=None, requested_end_s=None):
    stamps = [r["bag_timestamp_ns"] for r in records]
    diffs = [(b-a)/1e6 for a, b in zip(stamps, stamps[1:])]
    return {"name": sequence, "frames": len(records),
            "requested_start_s": requested_start_s,
            "requested_end_s": requested_end_s,
            "actual_first_s": records[0]["offset_s"] if records else None,
            "actual_last_s": records[-1]["offset_s"] if records else None,
            "median_frame_interval_ms": round(statistics.median(diffs), 3) if diffs else None,
            "max_frame_interval_ms": round(max(diffs), 3) if diffs else None}


def extract_one(source, out, name):
    bag = source / name
    metadata_path = bag / "metadata.yaml"
    info = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))[
        "rosbag2_bagfile_information"]
    image_topics = [t for t in topic_info({"rosbag2_bagfile_information": info})
                    if t["type"] == "sensor_msgs/msg/Image"]
    if len(image_topics) != 1 or image_topics[0]["name"] != "/infrared/image_raw":
        raise ValueError(f"Unexpected topics in {bag}: {image_topics}")
    storage_files = [p for p in bag.iterdir() if p.suffix in (".mcap", ".db3")]
    if len(storage_files) != 1:
        raise ValueError(f"Expected one bag storage file in {bag}")
    storage = storage_files[0]
    result = {"source_bag": str(bag), "storage": info["storage_identifier"],
              "storage_filename": storage.name, "storage_size_bytes": storage.stat().st_size,
              "metadata_sha256": sha256_file(metadata_path),
              "topic": image_topics[0], "duration_s": info["duration"]["nanoseconds"]/1e9,
              "sequences": []}
    all_records = []
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic == "/infrared/image_raw"]
        bag_start = reader.start_time
        duration = info["duration"]["nanoseconds"]
        interval_s = PLAN[name]["overview_interval_s"]
        overview_name = f"{name}/overview_every_{interval_s}s"
        overview = []
        for index, seconds in enumerate(range(0, int(duration/1e9), interval_s)):
            start = bag_start + seconds*1_000_000_000
            sample = next(reader.messages(connections=connections, start=start,
                                          stop=start+1_000_000_000), None)
            if sample is None:
                raise RuntimeError(f"No image near {seconds}s in {bag}")
            conn, stamp, raw = sample
            record = save_frame(out, overview_name, index, bag_start, stamp,
                                reader.deserialize(raw, conn.msgtype))
            overview.append(record)
        all_records.extend(overview)
        result["sequences"].append(describe_sequence(overview_name, overview))
        print(f"{name}: overview {len(overview)}", flush=True)

        for fraction in PLAN[name]["dense_fractions"]:
            sequence = f"{name}/clip_{int(fraction*100):02d}pct"
            start = bag_start + int(duration*fraction)
            records = []
            for conn, stamp, raw in reader.messages(connections=connections,
                                                    start=start,
                                                    stop=start+DENSE_DURATION_NS):
                records.append(save_frame(out, sequence, len(records), bag_start,
                                          stamp, reader.deserialize(raw, conn.msgtype)))
            if len(records) < 100 or len(records) > 160:
                raise RuntimeError(f"Unexpected six-second clip length {len(records)}: {bag}")
            all_records.extend(records)
            result["sequences"].append(describe_sequence(
                sequence, records, round((start-bag_start)/1e9, 3),
                round((start+DENSE_DURATION_NS-bag_start)/1e9, 3)))
            print(f"{name}: {sequence} {len(records)}", flush=True)
    with (out / name / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_records)
    result["exported_frames"] = len(all_records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not args.root.is_dir() or args.out.exists():
        raise ValueError("Source root must exist and output directory must be new")
    args.out.mkdir(parents=True)
    manifest = {"format_version": 1, "source_root": str(args.root),
                "extraction": "Y byte of UYVY YUV422 to lossless 8-bit PNG; no tonal change",
                "dense_clip_seconds": DENSE_DURATION_NS/1e9, "bags": []}
    for name in PLAN:
        record = extract_one(args.root, args.out, name)
        manifest["bags"].append(record)
        (args.out / "manifest.partial.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                 indent=2), encoding="utf-8")
    (args.out / "manifest.partial.json").unlink()
    print("complete", sum(x["exported_frames"] for x in manifest["bags"]), flush=True)


if __name__ == "__main__":
    main()
