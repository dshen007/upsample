"""Sample image messages from local ROS2 bags without changing the source bags.

Run under WSL: python3 inspect_flight_rosbags.py --root /home/.../flight_records
Requires rosbags, numpy, OpenCV, PyYAML. Only a few PNGs and JSON are written.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


NAMES = ("infrared_20260916_111549", "infrared_20260916_112100", "infrared_bag")


def image_array(message):
    height, width, step = int(message.height), int(message.width), int(message.step)
    encoding = message.encoding.lower()
    if encoding == "mono8":
        dtype, channels = np.uint8, 1
    elif encoding == "mono16":
        dtype, channels = np.dtype(">u2" if message.is_bigendian else "<u2"), 1
    elif encoding in ("yuv422", "yuv422_yuy2"):
        # ROS Humble image_encodings.hpp: yuv422 is UYVY, yuv422_yuy2 is YUYV.
        # The Y bytes are the unconverted camera luminance samples.
        dtype, channels = np.uint8, 2
    elif encoding in ("rgb8", "bgr8"):
        dtype, channels = np.uint8, 3
    else:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    raw = np.frombuffer(message.data, dtype=dtype)
    itemsize = np.dtype(dtype).itemsize
    if height <= 0 or width <= 0 or step < width * channels * itemsize:
        raise ValueError("Invalid image dimensions or row stride")
    if len(raw) != height * step // itemsize:
        raise ValueError("Unexpected image payload length")
    rows = raw.reshape(height, step // itemsize)
    pixels = rows[:, :width * channels]
    if encoding in ("yuv422", "yuv422_yuy2"):
        return pixels[:, 1::2].copy() if encoding == "yuv422" else pixels[:, 0::2].copy()
    if channels == 1:
        return pixels.reshape(height, width).astype(np.uint16 if itemsize == 2 else np.uint8)
    pixels = pixels.reshape(height, width, channels)
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY if encoding == "rgb8"
                        else cv2.COLOR_BGR2GRAY)


def topic_info(metadata):
    records = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]
    return [{"name": record["topic_metadata"]["name"],
             "type": record["topic_metadata"]["type"],
             "count": record["message_count"]} for record in records]


def inspect_one(root: Path, out: Path, name: str):
    bag = root / name
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())
    info = metadata["rosbag2_bagfile_information"]
    topics = topic_info(metadata)
    image_topics = [entry for entry in topics if entry["type"] == "sensor_msgs/msg/Image"]
    if len(image_topics) != 1:
        raise ValueError(f"Expected exactly one image topic in {name}: {image_topics}")
    topic = image_topics[0]
    result = {"bag": str(bag), "storage": info["storage_identifier"],
              "duration_s": info["duration"]["nanoseconds"] / 1e9,
              "image_topic": topic, "samples": [], "initial_frame_intervals_ms": []}
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic == topic["name"]]
        if not connections:
            raise ValueError(f"Image connection missing in {name}")
        times = []
        for conn, stamp, _ in reader.messages(connections=connections,
                                              stop=reader.start_time + 3_000_000_000):
            times.append(stamp)
            if len(times) >= 30:
                break
        result["initial_frame_intervals_ms"] = [round((b-a)/1e6, 3)
                                                 for a, b in zip(times, times[1:])]
        duration_ns = info["duration"]["nanoseconds"]
        for fraction in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9):
            start = reader.start_time + int(duration_ns * fraction)
            sample = next(reader.messages(connections=connections, start=start,
                                          stop=start + 2_000_000_000), None)
            if sample is None:
                result["samples"].append({"fraction": fraction, "found": False})
                continue
            conn, stamp, raw = sample
            message = reader.deserialize(raw, conn.msgtype)
            pixels = image_array(message)
            filename = f"sample_{round(fraction*100):02d}.png"
            if not cv2.imwrite(str(dest / filename), pixels):
                raise OSError(f"Could not write {dest / filename}")
            result["samples"].append({"fraction": fraction, "found": True,
                                      "offset_s": round((stamp-reader.start_time)/1e9, 3),
                                      "width": int(message.width), "height": int(message.height),
                                      "step": int(message.step), "encoding": message.encoding,
                                      "min": int(pixels.min()), "max": int(pixels.max()),
                                      "std": round(float(pixels.std()), 2),
                                      "file": str(dest / filename)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Use a new output directory: {args.out}")
    args.out.mkdir(parents=True)
    report = {"source_root": str(args.root), "bags": []}
    for name in NAMES:
        item = inspect_one(args.root, args.out, name)
        report["bags"].append(item)
        (args.out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False,
                                                       indent=2), encoding="utf-8")
        print(name, item["image_topic"], item["samples"][0] if item["samples"] else None,
              flush=True)


if __name__ == "__main__":
    main()
