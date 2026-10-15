"""PIE dataset analysis for graph construction.

解析 PIE 注释的三类 XML:
  1. annotations/<set>/<video>_annt.xml   -> 逐帧对象 track (pedestrian, vehicle,
     traffic_light, sign, crosswalk, transit_station)
  2. annotations_attributes/<set>/<video>_attributes.xml -> 行人级属性
     (age, gender, intention_prob, crossing, intersection, signalized, ...)
  3. annotations_vehicle/<set>/<video>_obd.xml -> 每帧自车 OBD / GPS

重点分析: intention, action, look; 其他 (gesture / cross / occlusion /
vehicle / traffic_light / sign / crosswalk / ego vehicle) 同步给出。

用法:
    python analysis/analyze_pie_dataset.py \
        --root /path/to/PIE_ROOT \
        --out analysis/results

输出:
    analysis/results/summary.json                  总体汇总 (机器可读)
    analysis/results/report.md                     人类可读报告
    analysis/results/ped_tracks.csv                每条行人 track 明细
    analysis/results/ped_attributes.csv            行人级属性明细
    analysis/results/label_distributions.csv       关键标签分布
    analysis/results/plots/*.png                   可视化图 (如装了 matplotlib)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class PedTrack:
    set_id: str
    video_id: str
    ped_id: str
    frames: List[int] = field(default_factory=list)
    action: List[str] = field(default_factory=list)
    look: List[str] = field(default_factory=list)
    cross: List[str] = field(default_factory=list)
    gesture: List[str] = field(default_factory=list)
    occlusion: List[str] = field(default_factory=list)
    bbox: List[Tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class PedAttr:
    set_id: str
    video_id: str
    ped_id: str
    age: str = ""
    gender: str = ""
    crossing: int = 0  # 1 yes / 0 no / -1 irrelevant
    intention_prob: float = float("nan")
    critical_point: int = -1
    crossing_point: int = -1
    exp_start_point: int = -1
    intersection: str = ""
    signalized: str = ""
    num_lanes: int = -1
    traffic_direction: str = ""


# --------------------------------------------------------------------------- #
# XML parsing
# --------------------------------------------------------------------------- #
def _attrs_of_box(box: ET.Element) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for a in box.findall("attribute"):
        out[a.attrib.get("name", "")] = (a.text or "").strip()
    return out


def parse_annt_xml(path: Path) -> Tuple[Dict[str, PedTrack], Dict[str, Counter], Dict]:
    """Parse one <video>_annt.xml.

    Returns (ped_tracks, other_counts, meta) where other_counts has keys
    vehicle/sign/traffic_light/crosswalk/transit_station each mapping
    (type or state) -> frame count, plus track-level stats.
    """
    set_id = path.parent.name
    video_id = path.stem.replace("_annt", "")
    tree = ET.parse(path)
    root = tree.getroot()

    size_el = root.find(".//task/size")
    n_frames = int(size_el.text) if size_el is not None and size_el.text else -1

    ped_tracks: Dict[str, PedTrack] = {}
    other_counts: Dict[str, Counter] = {
        "vehicle_type": Counter(),
        "sign_type": Counter(),
        "tl_state": Counter(),
        "tl_type": Counter(),
        "crosswalk_frames": Counter(),  # just "count"
        "transit_station_frames": Counter(),
    }
    track_counts = Counter()

    for track in root.findall("track"):
        label = track.attrib.get("label", "")
        track_counts[label] += 1
        boxes = track.findall("box")
        if not boxes:
            continue

        if label == "pedestrian":
            # pedestrian id comes from attribute
            first_attrs = _attrs_of_box(boxes[0])
            pid = first_attrs.get("id", f"{set_id}_{video_id}_unknown")
            pt = ped_tracks.setdefault(
                pid, PedTrack(set_id=set_id, video_id=video_id, ped_id=pid)
            )
            for b in boxes:
                if int(b.attrib.get("outside", "0")) == 1:
                    continue
                a = _attrs_of_box(b)
                pt.frames.append(int(b.attrib["frame"]))
                pt.action.append(a.get("action", ""))
                pt.look.append(a.get("look", ""))
                pt.cross.append(a.get("cross", ""))
                pt.gesture.append(a.get("gesture", ""))
                pt.occlusion.append(a.get("occlusion", ""))
                pt.bbox.append(
                    (
                        float(b.attrib.get("xtl", 0)),
                        float(b.attrib.get("ytl", 0)),
                        float(b.attrib.get("xbr", 0)),
                        float(b.attrib.get("ybr", 0)),
                    )
                )
        elif label == "vehicle":
            for b in boxes:
                if int(b.attrib.get("outside", "0")) == 1:
                    continue
                a = _attrs_of_box(b)
                other_counts["vehicle_type"][a.get("type", "unk")] += 1
        elif label == "sign":
            for b in boxes:
                if int(b.attrib.get("outside", "0")) == 1:
                    continue
                a = _attrs_of_box(b)
                other_counts["sign_type"][a.get("type", "unk")] += 1
        elif label == "traffic_light":
            for b in boxes:
                if int(b.attrib.get("outside", "0")) == 1:
                    continue
                a = _attrs_of_box(b)
                other_counts["tl_state"][a.get("state", "unk")] += 1
                other_counts["tl_type"][a.get("type", "unk")] += 1
        elif label == "crosswalk":
            cnt = sum(1 for b in boxes if int(b.attrib.get("outside", "0")) == 0)
            other_counts["crosswalk_frames"]["total"] += cnt
        elif label == "transit_station":
            cnt = sum(1 for b in boxes if int(b.attrib.get("outside", "0")) == 0)
            other_counts["transit_station_frames"]["total"] += cnt

    meta = {
        "set_id": set_id,
        "video_id": video_id,
        "n_frames": n_frames,
        "track_counts": dict(track_counts),
    }
    return ped_tracks, other_counts, meta


def parse_attr_xml(path: Path) -> List[PedAttr]:
    set_id = path.parent.name
    video_id = path.stem.replace("_attributes", "")
    tree = ET.parse(path)
    out: List[PedAttr] = []
    for p in tree.getroot().findall("pedestrian"):
        a = p.attrib
        try:
            ip = float(a.get("intention_prob", "nan"))
        except ValueError:
            ip = float("nan")
        try:
            nl = int(a.get("num_lanes", "-1"))
        except ValueError:
            nl = -1
        try:
            crossing = int(a.get("crossing", "0"))
        except ValueError:
            crossing = 0
        out.append(
            PedAttr(
                set_id=set_id,
                video_id=video_id,
                ped_id=a.get("id", ""),
                age=a.get("age", ""),
                gender=a.get("gender", ""),
                crossing=crossing,
                intention_prob=ip,
                critical_point=int(a.get("critical_point", -1) or -1),
                crossing_point=int(a.get("crossing_point", -1) or -1),
                exp_start_point=int(a.get("exp_start_point", -1) or -1),
                intersection=a.get("intersection", ""),
                signalized=a.get("signalized", ""),
                num_lanes=nl,
                traffic_direction=a.get("traffic_direction", ""),
            )
        )
    return out


def parse_vehicle_xml(path: Path) -> Dict:
    """Aggregate per-frame OBD statistics for one video."""
    tree = ET.parse(path)
    frames = tree.getroot().findall("frame")
    gps = [float(f.attrib.get("GPS_speed", 0)) for f in frames]
    obd = [float(f.attrib.get("OBD_speed", 0)) for f in frames]
    accY = [float(f.attrib.get("accY", 0)) for f in frames]
    yaw_rate = [float(f.attrib.get("gyroZ", 0)) for f in frames]

    def stats(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": mean(xs),
            "median": median(xs),
            "std": pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }

    return {
        "n_frames": len(frames),
        "GPS_speed": stats(gps),
        "OBD_speed": stats(obd),
        "accY": stats(accY),
        "gyroZ": stats(yaw_rate),
    }


# --------------------------------------------------------------------------- #
# Track-level derived stats
# --------------------------------------------------------------------------- #
def _switches(seq: List[str]) -> int:
    return sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)


def _longest_run(seq: List[str], target: str) -> int:
    best = cur = 0
    for s in seq:
        if s == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def summarize_track(pt: PedTrack) -> Dict:
    L = len(pt.frames)
    return {
        "set_id": pt.set_id,
        "video_id": pt.video_id,
        "ped_id": pt.ped_id,
        "length": L,
        "n_action_switch": _switches(pt.action),
        "n_look_switch": _switches(pt.look),
        "n_cross_switch": _switches(pt.cross),
        "n_occlusion_switch": _switches(pt.occlusion),
        "frac_looking": pt.look.count("looking") / L if L else 0.0,
        "frac_walking": pt.action.count("walking") / L if L else 0.0,
        "frac_crossing": pt.cross.count("crossing") / L if L else 0.0,
        "frac_occluded_any": sum(1 for o in pt.occlusion if o in ("part", "full")) / L
        if L
        else 0.0,
        "longest_look_run": _longest_run(pt.look, "looking"),
        "longest_notlook_run": _longest_run(pt.look, "not-looking"),
        "longest_walk_run": _longest_run(pt.action, "walking"),
        "longest_stand_run": _longest_run(pt.action, "standing"),
        "has_gesture": any(
            g not in ("", "__undefined__") for g in pt.gesture
        ),
    }


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def analyze(root: Path, out_dir: Path) -> None:
    ann_root = root / "annotations" / "annotations"
    attr_root = root / "annotations" / "annotations_attributes"
    veh_root = root / "annotations" / "annotations_vehicle"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    all_ped_tracks: List[PedTrack] = []
    video_metas: List[Dict] = []
    other_totals: Dict[str, Counter] = defaultdict(Counter)
    veh_stats: List[Dict] = []

    annt_files = sorted(ann_root.glob("set*/video_*_annt.xml"))
    if not annt_files:
        print(f"[error] no annt files under {ann_root}", file=sys.stderr)
        sys.exit(1)
    print(f"[info] parsing {len(annt_files)} annt xml files ...")
    for f in annt_files:
        tracks, others, meta = parse_annt_xml(f)
        all_ped_tracks.extend(tracks.values())
        for k, c in others.items():
            other_totals[k].update(c)
        video_metas.append(meta)

    attr_files = sorted(attr_root.glob("set*/video_*_attributes.xml"))
    all_attrs: List[PedAttr] = []
    print(f"[info] parsing {len(attr_files)} attribute xml files ...")
    for f in attr_files:
        all_attrs.extend(parse_attr_xml(f))

    veh_files = sorted(veh_root.glob("set*/video_*_obd.xml"))
    print(f"[info] parsing {len(veh_files)} vehicle obd xml files ...")
    for f in veh_files:
        s = parse_vehicle_xml(f)
        s["set_id"] = f.parent.name
        s["video_id"] = f.stem.replace("_obd", "")
        veh_stats.append(s)

    # --- per-frame label counts over all pedestrians ---
    frame_counts = {
        "action": Counter(),
        "look": Counter(),
        "cross": Counter(),
        "gesture": Counter(),
        "occlusion": Counter(),
    }
    for pt in all_ped_tracks:
        frame_counts["action"].update(pt.action)
        frame_counts["look"].update(pt.look)
        frame_counts["cross"].update(pt.cross)
        frame_counts["gesture"].update(pt.gesture)
        frame_counts["occlusion"].update(pt.occlusion)

    # --- per-track summary ---
    track_rows = [summarize_track(pt) for pt in all_ped_tracks]

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return mean(xs) if xs else 0.0

    track_summary = {
        "n_tracks": len(track_rows),
        "mean_length": _mean([r["length"] for r in track_rows]),
        "median_length": median([r["length"] for r in track_rows]) if track_rows else 0,
        "mean_look_switch": _mean([r["n_look_switch"] for r in track_rows]),
        "mean_action_switch": _mean([r["n_action_switch"] for r in track_rows]),
        "mean_cross_switch": _mean([r["n_cross_switch"] for r in track_rows]),
        "frac_tracks_any_look": sum(1 for r in track_rows if r["frac_looking"] > 0)
        / max(1, len(track_rows)),
        "frac_tracks_pure_notlook": sum(1 for r in track_rows if r["frac_looking"] == 0)
        / max(1, len(track_rows)),
        "frac_tracks_pure_look": sum(1 for r in track_rows if r["frac_looking"] == 1)
        / max(1, len(track_rows)),
        "mean_longest_look_run": _mean([r["longest_look_run"] for r in track_rows]),
        "mean_longest_notlook_run": _mean([r["longest_notlook_run"] for r in track_rows]),
    }

    # --- pedestrian-level attributes ---
    attr_dist = {
        "age": Counter(a.age for a in all_attrs),
        "gender": Counter(a.gender for a in all_attrs),
        "crossing": Counter(a.crossing for a in all_attrs),
        "intersection": Counter(a.intersection for a in all_attrs),
        "signalized": Counter(a.signalized for a in all_attrs),
        "num_lanes": Counter(a.num_lanes for a in all_attrs),
        "traffic_direction": Counter(a.traffic_direction for a in all_attrs),
    }
    intents = [a.intention_prob for a in all_attrs if a.intention_prob == a.intention_prob]
    intent_stats = {
        "n": len(intents),
        "mean": mean(intents) if intents else 0,
        "median": median(intents) if intents else 0,
        "std": pstdev(intents) if len(intents) > 1 else 0,
        "hist_bins": [0, 0.2, 0.4, 0.6, 0.8, 1.0001],
        "hist_counts": None,
    }
    if intents:
        bins = intent_stats["hist_bins"]
        hc = [0] * (len(bins) - 1)
        for v in intents:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    hc[i] += 1
                    break
        intent_stats["hist_counts"] = hc

    # --- write csvs ---
    import csv

    with (out_dir / "ped_tracks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()) if track_rows else [])
        w.writeheader()
        w.writerows(track_rows)

    with (out_dir / "ped_attributes.csv").open("w", newline="") as f:
        fields = [
            "set_id", "video_id", "ped_id", "age", "gender", "crossing",
            "intention_prob", "critical_point", "crossing_point",
            "exp_start_point", "intersection", "signalized", "num_lanes",
            "traffic_direction",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in all_attrs:
            w.writerow(a.__dict__)

    with (out_dir / "label_distributions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "label", "count"])
        for g, c in frame_counts.items():
            for k, v in c.most_common():
                w.writerow([g, k, v])
        for g, c in other_totals.items():
            for k, v in c.most_common():
                w.writerow([g, k, v])

    # --- plots ---
    if HAS_MPL:
        def bar(counter: Counter, title: str, fname: str):
            items = counter.most_common()
            if not items:
                return
            keys = [str(k) for k, _ in items]
            vals = [v for _, v in items]
            fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(keys) + 2), 3.2))
            ax.bar(range(len(keys)), vals)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=30, ha="right")
            ax.set_title(title)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / fname, dpi=140)
            plt.close(fig)

        for g, c in frame_counts.items():
            bar(c, f"Per-frame pedestrian label: {g}", f"frames_{g}.png")
        for g, c in other_totals.items():
            bar(c, f"Other object frames: {g}", f"other_{g}.png")
        for k, c in attr_dist.items():
            bar(c, f"Pedestrian attribute: {k}", f"attr_{k}.png")

        # look-switch histogram
        if track_rows:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            xs = [r["n_look_switch"] for r in track_rows]
            ax.hist(xs, bins=range(0, max(xs) + 2))
            ax.set_title("Per-track look switches")
            ax.set_xlabel("#switches")
            fig.tight_layout()
            fig.savefig(plot_dir / "look_switches.png", dpi=140)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 3.2))
            xs = [r["frac_looking"] for r in track_rows]
            ax.hist(xs, bins=20)
            ax.set_title("Per-track frac_looking")
            fig.tight_layout()
            fig.savefig(plot_dir / "frac_looking.png", dpi=140)
            plt.close(fig)

        # intention_prob hist
        if intents:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.hist(intents, bins=20)
            ax.set_title("Pedestrian intention_prob")
            fig.tight_layout()
            fig.savefig(plot_dir / "intention_prob.png", dpi=140)
            plt.close(fig)

    # --- vehicle ego summary ---
    def _agg(stats_list: List[Dict], key: str) -> Dict:
        vals_mean = [s[key]["mean"] for s in stats_list if s[key].get("n", 0) > 0]
        vals_max = [s[key]["max"] for s in stats_list if s[key].get("n", 0) > 0]
        return {
            "n_videos": len(vals_mean),
            "mean_of_means": mean(vals_mean) if vals_mean else 0,
            "max_of_max": max(vals_max) if vals_max else 0,
        }

    veh_summary = {
        "GPS_speed": _agg(veh_stats, "GPS_speed"),
        "OBD_speed": _agg(veh_stats, "OBD_speed"),
        "accY": _agg(veh_stats, "accY"),
        "gyroZ": _agg(veh_stats, "gyroZ"),
        "total_frames": sum(s["n_frames"] for s in veh_stats),
    }

    # --- summary json ---
    summary = {
        "n_videos": len(video_metas),
        "n_sets": len({m["set_id"] for m in video_metas}),
        "total_video_frames": sum(m["n_frames"] for m in video_metas if m["n_frames"] > 0),
        "n_pedestrian_tracks": len(all_ped_tracks),
        "n_pedestrian_attrs": len(all_attrs),
        "frame_counts": {k: dict(v) for k, v in frame_counts.items()},
        "other_object_counts": {k: dict(v) for k, v in other_totals.items()},
        "track_summary": track_summary,
        "ped_attr_distribution": {k: {str(kk): vv for kk, vv in v.items()} for k, v in attr_dist.items()},
        "intention_prob": intent_stats,
        "vehicle_ego_summary": veh_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # --- markdown report ---
    md = _render_report(summary)
    (out_dir / "report.md").write_text(md)
    print(f"[done] results -> {out_dir}")


def _render_report(s: Dict) -> str:
    def _fmt_counter(d: Dict, top: int = 20) -> str:
        items = sorted(d.items(), key=lambda x: -x[1])[:top]
        total = sum(d.values()) or 1
        lines = ["| 类别 | 计数 | 占比 |", "|---|---:|---:|"]
        for k, v in items:
            lines.append(f"| {k} | {v} | {v/total*100:.2f}% |")
        return "\n".join(lines)

    ts = s["track_summary"]
    ip = s["intention_prob"]
    md = []
    md.append("# PIE 数据集图构建前置分析\n")
    md.append(
        f"- 视频数 (annt xml): **{s['n_videos']}**，划分 set 数: **{s['n_sets']}**，"
        f"视频总帧 meta: **{s['total_video_frames']}**\n"
        f"- 行人 track 数: **{s['n_pedestrian_tracks']}**，行人属性条目数 "
        f"(annotations_attributes): **{s['n_pedestrian_attrs']}**\n"
    )

    md.append("## 1. 逐帧行人标签分布 (所有 pedestrian track 的 box 聚合)\n")
    for g in ("action", "look", "cross", "gesture", "occlusion"):
        md.append(f"### {g}\n")
        md.append(_fmt_counter(s["frame_counts"].get(g, {})))
        md.append("\n")

    md.append("## 2. 行人 track 级结构 (重点: look / action 是否\"瞬时信号\")\n")
    md.append(
        f"- 行人 track 数: **{ts['n_tracks']}**\n"
        f"- 平均 track 长度 (帧): **{ts['mean_length']:.1f}**，中位数: **{ts['median_length']}**\n"
        f"- 平均每 track look 切换次数: **{ts['mean_look_switch']:.2f}**\n"
        f"- 平均每 track action 切换次数: **{ts['mean_action_switch']:.2f}**\n"
        f"- 平均每 track cross 切换次数: **{ts['mean_cross_switch']:.2f}**\n"
        f"- 全程从未 looking 的 track 占比: **{ts['frac_tracks_pure_notlook']*100:.2f}%**\n"
        f"- 全程都在 looking 的 track 占比: **{ts['frac_tracks_pure_look']*100:.2f}%**\n"
        f"- 至少出现一次 looking 的 track 占比: **{ts['frac_tracks_any_look']*100:.2f}%**\n"
        f"- 平均最长连续 looking 段: **{ts['mean_longest_look_run']:.1f}** 帧；"
        f"最长连续 not-looking 段: **{ts['mean_longest_notlook_run']:.1f}** 帧\n"
        "\n> 结论模版: 若 `mean_look_switch` 很小而 `mean_longest_look_run`/平均长度 接近 1,"
        " 说明 look 更接近**行人整体状态**而非瞬时事件, 图建模时可以把 look "
        "视作 track 级弱标签而非每帧独立监督.\n"
    )

    md.append("## 3. intention 分析 (annotations_attributes.intention_prob)\n")
    md.append(
        f"- n={ip['n']}, mean={ip['mean']:.3f}, median={ip['median']:.3f}, std={ip['std']:.3f}\n"
    )
    if ip.get("hist_counts"):
        md.append("- 分桶 [0,0.2),[0.2,0.4),[0.4,0.6),[0.6,0.8),[0.8,1.0]:\n")
        md.append(f"  - {ip['hist_counts']}\n")
    md.append("\n### 3.1 crossing 真实标签分布 (1=cross, 0=not-cross, -1=irrelevant)\n")
    md.append(_fmt_counter(s["ped_attr_distribution"].get("crossing", {})))
    md.append("\n")

    md.append("## 4. 行人人口与场景属性分布\n")
    for k in ("age", "gender", "intersection", "signalized", "num_lanes", "traffic_direction"):
        md.append(f"### {k}\n")
        md.append(_fmt_counter(s["ped_attr_distribution"].get(k, {})))
        md.append("\n")

    md.append("## 5. 其他对象 (用于构图的非行人节点)\n")
    for g in ("vehicle_type", "sign_type", "tl_state", "tl_type",
              "crosswalk_frames", "transit_station_frames"):
        md.append(f"### {g}\n")
        md.append(_fmt_counter(s["other_object_counts"].get(g, {})))
        md.append("\n")

    veh = s["vehicle_ego_summary"]
    md.append("## 6. 自车 OBD / GPS (ego 节点特征)\n")
    md.append(
        f"- 总 OBD 帧数: {veh['total_frames']}\n"
        f"- GPS_speed: mean-of-means={veh['GPS_speed']['mean_of_means']:.3f}, max-of-max={veh['GPS_speed']['max_of_max']:.3f}\n"
        f"- OBD_speed: mean-of-means={veh['OBD_speed']['mean_of_means']:.3f}, max-of-max={veh['OBD_speed']['max_of_max']:.3f}\n"
        f"- accY: mean-of-means={veh['accY']['mean_of_means']:.3f}, max-of-max={veh['accY']['max_of_max']:.3f}\n"
        f"- gyroZ (yaw-rate): mean-of-means={veh['gyroZ']['mean_of_means']:.3f}, max-of-max={veh['gyroZ']['max_of_max']:.3f}\n"
    )

    md.append("## 7. 对图构建的启示\n")
    md.append(
        "- **节点类型**: pedestrian / vehicle / traffic_light / sign / crosswalk /"
        " transit_station / ego (自车)。可按上面 \"其他对象\" 分布评估每类节点的稀疏度。\n"
        "- **标签粒度**: `look` 切换率低 -> 建议作为 track 级状态或 slow-changing 特征;"
        " `action` 切换率同理; `cross/intention` 则更偏事件型, 宜做时序监督。\n"
        "- **正负样本失衡**: 根据 crossing=1/0/-1 分布和 intention_prob 分桶, 训练时建议用"
        " focal loss 或按 per-video 重采样。\n"
        "- **场景上下文**: intersection / signalized / num_lanes / traffic_direction"
        " 可直接作为 pedestrian static feature 或 global 条件 (FiLM)。\n"
        "- **ego 特征**: GPS_speed / OBD_speed / accY / gyroZ 的量级差异大, 建图前需做"
        " per-channel 标准化。\n"
    )
    return "\n".join(md)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="PIE workspace root (contains annotations/)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    args = ap.parse_args()
    analyze(Path(args.root), Path(args.out))
