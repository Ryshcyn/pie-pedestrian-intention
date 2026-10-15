"""Build the processed PIE graph dataset used by the final experiments.

The raw PIE dataset is intentionally not bundled with this repository. Download it
from the official source, then point ``--pie-root`` at the directory containing
``annotations/``, ``annotations_attributes/`` and ``annotations_vehicle/``.
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

TRACK_TYPES = ("pedestrian", "vehicle", "traffic_light", "crosswalk")


def parse_annotation_xml(xml_path: Path) -> dict:
    tracks = {kind: [] for kind in TRACK_TYPES}
    for track_el in ET.parse(xml_path).getroot().findall("track"):
        label = track_el.get("label")
        if label not in tracks:
            continue
        boxes = []
        for box_el in track_el.findall("box"):
            if int(box_el.get("outside", "0")) == 1:
                continue
            attrs = {
                attr.get("name"): (attr.text or "")
                for attr in box_el.findall("attribute")
            }
            boxes.append({
                "frame": int(box_el.get("frame")),
                "xtl": float(box_el.get("xtl")),
                "ytl": float(box_el.get("ytl")),
                "xbr": float(box_el.get("xbr")),
                "ybr": float(box_el.get("ybr")),
                "attrs": attrs,
            })
        if boxes:
            tracks[label].append(boxes)
    return tracks


def parse_attributes_xml(xml_path: Path) -> dict:
    """Return every field used by intention labels and static scene context."""
    if not xml_path.exists():
        return {}
    result = {}
    for ped_el in ET.parse(xml_path).getroot().findall("pedestrian"):
        pid = ped_el.get("id")
        if not pid:
            continue
        try:
            intention_prob = float(ped_el.get("intention_prob", "-1"))
        except ValueError:
            intention_prob = -1.0
        result[pid] = {
            "intention_prob": intention_prob,
            "intersection": ped_el.get("intersection", ""),
            "signalized": ped_el.get("signalized", "n/a"),
            "num_lanes": ped_el.get("num_lanes", "0"),
        }
    return result


def parse_obd_xml(xml_path: Path) -> dict:
    """Return frame dictionaries; graph_builder consumes ``OBD_speed`` by name."""
    if not xml_path.exists():
        return {}
    result = {}
    for frame_el in ET.parse(xml_path).getroot().findall("frame"):
        try:
            speed = float(frame_el.get("OBD_speed", "0"))
        except ValueError:
            speed = 0.0
        result[int(frame_el.get("id"))] = {"OBD_speed": speed}
    return result


def build_frame_index(tracks: dict) -> dict:
    frames = defaultdict(lambda: {
        "pedestrians": [], "vehicles": [],
        "traffic_lights": [], "crosswalks": [],
    })
    for track_boxes in tracks["pedestrian"]:
        for box in track_boxes:
            attrs = box["attrs"]
            frames[box["frame"]]["pedestrians"].append({
                "id": attrs.get("id", ""),
                "xtl": box["xtl"], "ytl": box["ytl"],
                "xbr": box["xbr"], "ybr": box["ybr"],
                "occlusion": attrs.get("occlusion", "none"),
                "action": attrs.get("action", "standing"),
                "look": attrs.get("look", "not-looking"),
            })
    for track_boxes in tracks["vehicle"]:
        for box in track_boxes:
            attrs = box["attrs"]
            frames[box["frame"]]["vehicles"].append({
                "xtl": box["xtl"], "ytl": box["ytl"],
                "xbr": box["xbr"], "ybr": box["ybr"],
                "type": attrs.get("type", "car"),
            })
    for track_boxes in tracks["traffic_light"]:
        for box in track_boxes:
            attrs = box["attrs"]
            frames[box["frame"]]["traffic_lights"].append({
                "xtl": box["xtl"], "ytl": box["ytl"],
                "xbr": box["xbr"], "ybr": box["ybr"],
                "state": attrs.get("state", "__undefined__"),
                "type": attrs.get("type", "regular"),
            })
    for track_boxes in tracks["crosswalk"]:
        for box in track_boxes:
            frames[box["frame"]]["crosswalks"].append({
                "xtl": box["xtl"], "ytl": box["ytl"],
                "xbr": box["xbr"], "ybr": box["ybr"],
            })
    return dict(frames)


def get_all_videos(annotation_dir: Path) -> list[tuple[str, str]]:
    videos = []
    for xml_path in sorted(annotation_dir.glob("set*/*_annt.xml")):
        videos.append((xml_path.parent.name, xml_path.name.removesuffix("_annt.xml")))
    return videos


def split_videos(videos, seed: int, train_ratio: float, val_ratio: float) -> dict:
    shuffled = list(videos)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def enumerate_windows(video_db: dict, t_obs: int, stride: int) -> list[tuple[int, int]]:
    frame_objects = video_db["frame_objects"]
    ped_frames = sorted(fid for fid, data in frame_objects.items() if data["pedestrians"])
    windows = []
    for end in ped_frames[::stride]:
        start = end - t_obs + 1
        common_pids = None
        for fid in range(start, end + 1):
            frame = frame_objects.get(fid)
            if frame is None:
                common_pids = set()
                break
            pids = {ped["id"] for ped in frame["pedestrians"]}
            common_pids = pids if common_pids is None else common_pids & pids
            if not common_pids:
                break
        if common_pids:
            windows.append((start, end))
    return windows


def compute_label_stats(databases: dict, splits: dict, windows_by_video: dict) -> dict:
    stats = {}
    for split, videos in splits.items():
        counts = dict(int_pos=0, int_neg=0, int_na=0,
                      act_pos=0, act_neg=0, n_samples=0)
        for set_id, video_id in videos:
            key = f"{set_id}/{video_id}"
            db = databases[key]
            for _, end in windows_by_video[key]:
                counts["n_samples"] += 1
                for ped in db["frame_objects"].get(end, {}).get("pedestrians", []):
                    counts["act_pos" if ped["action"] == "walking" else "act_neg"] += 1
                    attrs = db["ped_attributes"].get(ped["id"])
                    if attrs is None or attrs["intention_prob"] < 0:
                        counts["int_na"] += 1
                    else:
                        counts["int_pos" if attrs["intention_prob"] > 0.5 else "int_neg"] += 1
        stats[split] = counts
    return stats


def process_video(pie_root: Path, set_id: str, video_id: str) -> dict:
    tracks = parse_annotation_xml(
        pie_root / "annotations" / set_id / f"{video_id}_annt.xml")
    return {
        "frame_objects": build_frame_index(tracks),
        "ped_attributes": parse_attributes_xml(
            pie_root / "annotations_attributes" / set_id / f"{video_id}_attributes.xml"),
        "ego_data": parse_obd_xml(
            pie_root / "annotations_vehicle" / set_id / f"{video_id}_obd.xml"),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--pie-root", required=True, type=Path)
    parser.add_argument("--output", default="data/processed", type=Path)
    parser.add_argument("--t-obs", default=15, type=int)
    parser.add_argument("--stride", default=15, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train-ratio", default=0.70, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    args = parser.parse_args()

    videos = get_all_videos(args.pie_root / "annotations")
    if not videos:
        raise SystemExit("No *_annt.xml files found under --pie-root/annotations")

    databases = {}
    for index, (set_id, video_id) in enumerate(videos, start=1):
        key = f"{set_id}/{video_id}"
        print(f"[{index}/{len(videos)}] {key}")
        databases[key] = process_video(args.pie_root, set_id, video_id)

    splits = split_videos(videos, args.seed, args.train_ratio, args.val_ratio)
    windows_by_video = {
        key: enumerate_windows(db, args.t_obs, args.stride)
        for key, db in databases.items()
    }
    split_windows = {
        split: [
            {"set_id": set_id, "vid_id": video_id,
             "frame_start": start, "frame_end": end}
            for set_id, video_id in video_list
            for start, end in windows_by_video[f"{set_id}/{video_id}"]
        ]
        for split, video_list in splits.items()
    }
    label_stats = compute_label_stats(databases, splits, windows_by_video)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(databases, args.output / "video_databases.pt")
    with open(args.output / "split_info.json", "w", encoding="utf-8") as handle:
        json.dump({
            "splits": splits,
            "windows": split_windows,
            "label_stats": label_stats,
        }, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: len(v) for k, v in split_windows.items()}, indent=2))
    print(f"Saved processed data to {args.output}")


if __name__ == "__main__":
    main()
