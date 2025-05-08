import os
import sys
import json
import argparse
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import nms
from lib.test.evaluation import get_dataset
from tqdm import tqdm
from typing import List
import math
import cv2

def load_candidate_boxes(step3_dir: str, seq_name: str, frame_idx: int) -> List[List[float]]:
    """从 step3_filtered 中加载指定帧的候选框列表，boxes[0] 为 GT 框"""
    path = os.path.join(step3_dir, f"{seq_name}.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到候选框文件: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if i == frame_idx:
                data = json.loads(line.strip())
                return data
    return []

def compute_iou(box1, box2):
    """计算两个框的 IOU"""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xi1, yi1 = max(x1, a1), max(y1, b1)
    xi2, yi2 = min(x2, a2), min(y2, b2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0



def detect_scene_change_gt(
    seq_name: str,
    idx1: int,
    idx2: int,
    step3_dir: str,
    # 宽松一点的阈值
    iou_thresh: float = 0.5,
    aspect_ratio_thresh: float = 0.8,
    center_dist_thresh: float = 0.4,
    small_area_thresh: float = 0.005
) -> bool:
    """
    更宽松的剧烈变化检测，减少 SOI 帧。
    """
    jsonl_path = os.path.join(step3_dir, f"{seq_name}.jsonl")
    if not os.path.isfile(jsonl_path):
        return True

    def read_boxes(idx: int):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                if i == idx:
                    return json.loads(line.strip())
        return []

    boxes1 = read_boxes(idx1)
    boxes2 = read_boxes(idx2)
    if not boxes1 or not boxes2:
        return True

    box1, box2 = boxes1[0], boxes2[0]

    # 小目标跳过
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    diag_sq = w1 * w1 + h1 * h1
    area1 = w1 * h1
    if diag_sq > 0 and (area1 / (diag_sq + 1e-6)) < small_area_thresh:
        return False

    # IoU
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = area1 + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / (union + 1e-6)
    if iou < iou_thresh:
        return True

    # 宽高比
    def aspect_ratio(box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        return w / h if h > 0 else 0
    ratio1, ratio2 = aspect_ratio(box1), aspect_ratio(box2)
    if abs(math.log(ratio1 + 1e-6) - math.log(ratio2 + 1e-6)) > aspect_ratio_thresh:
        return True

    # 中心点位移
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    center_dist = math.hypot(cx1 - cx2, cy1 - cy2) / (math.sqrt(diag_sq) + 1e-6)
    if center_dist > center_dist_thresh:
        return True

    return False



def detect_scene_change_soi(seq_name, idx1, idx2, step3_dir, iou_thresh=0.5, min_matched=1):
    """
    判断 SOI 候选框是否剧烈变化（匹配数不足）
    - 匹配逻辑：只要 frame1 中某个框在 frame2 中存在 IOU > 阈值 的配对，就视为匹配成功
    - 若匹配数 < min_matched，则认为场景发生了明显变化
    """
    try:
        jsonl_path = os.path.join(step3_dir, f"{seq_name}.jsonl")
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"候选框文件不存在: {jsonl_path}")

        def read_boxes(idx):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    if i == idx:
                        return json.loads(line.strip())[1:]  # 跳过 GT
            return []

        boxes1 = read_boxes(idx1)
        boxes2 = read_boxes(idx2)

        if not boxes1 or not boxes2:
            return True  # 任意一帧无框，默认视为剧烈变化

        matched = 0
        for b1 in boxes1:
            if any(compute_iou(b1, b2) > iou_thresh for b2 in boxes2):
                matched += 1
        return matched < min_matched

    except Exception as e:
        print(f"[SOI变化检测错误] {seq_name} idx1={idx1} idx2={idx2}：{e}")
        return True

def is_small_box(box: List[int], img_w: int, img_h: int,
                 area_thresh: float = 0.00005, min_w: int = 16, min_h: int = 16) -> bool:
    """
    判断一个框是否“过小”：
      - w*h / (img_w*img_h) < area_thresh
      - 或者 w < min_w, h < min_h
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return True
    area_ratio = (w * h) / (img_w * img_h)
    if area_ratio < area_thresh or w < min_w or h < min_h:
        return True
    return False

def extract_soi_frames(status_dir, step3_dir, save_dir, dataset_name="lasot",
                       frame_gap_threshold=10, iou_thresh_gt=0.7, iou_thresh_soi=0.5,
                       min_vote_ratio=0.5):
    """
    提取 SOI 描述帧：
    1️⃣ 状态投票得到初步 SOI 干扰帧
    2️⃣ 基于帧间隔和剧烈变化判断最终需生成文本描述的帧
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset = get_dataset(dataset_name)

    for seq in tqdm(dataset, desc="提取 SOI 描述帧"):
        seq_name = seq.name
        seq_len = len(seq.frames)
        trackers = []

        # ✅ 加载多个 tracker 的状态文件
        for tracker in os.listdir(status_dir):
            file = os.path.join(status_dir, tracker, f"{seq_name}_status.txt")
            if not os.path.isfile(file): continue
            with open(file, 'r') as f:
                lines = [l.strip() for l in f]
                if len(lines) == seq_len - 1:
                    trackers.append(lines)

        if not trackers:
            print(f"🚫 无状态文件: {seq_name}")
            continue

        # ✅ 阶段一：投票提取 SOI 候选帧
        soi_candidates = []
        for idx in range(seq_len - 1):
            votes = [t[idx] for t in trackers]
            if all(v == 'absent' for v in votes):
                continue
            if sum(seq.ground_truth_rect[idx]) == 0.:  # 跳过ground truth [0,0,0,0] 但在soi推理的时候没正确标注absent的情况
                continue
            ratio = sum(v in ('Drift', 'Fail') for v in votes) / len(votes)
            if ratio >= min_vote_ratio:  # 4个投票者，对半呗
                soi_candidates.append(idx + 1)

        # ✅ 阶段二：结合间隔和“剧烈变化”筛选需更新文本的帧
        final_frames = []
        last_idx = -frame_gap_threshold - 1
        for idx in soi_candidates:
            # 👇 加上这一步：跳过候选框为空的帧（只有GT）
            if idx==23 and seq_name=='licenseplate-13':
                print('debug')
            try:
                boxes = load_candidate_boxes(step3_dir, seq_name, idx)
                if len(boxes) <= 1:  # 只有 GT 或为空
                    continue
            except Exception as e:
                print(f"⚠️ 跳过 {seq_name} 第{idx}帧（读取候选框失败）: {e}")
                continue
            
            # ———— 如果 GT 本身过小，就直接跳过本帧 ————
            img_path = seq.frames[idx]
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            gt_box = boxes[0]

            if is_small_box(gt_box, img_w, img_h):
                print(f"⚠️ 跳过 {seq_name} 第{idx}帧（GT过小）")
                continue
            # ———————————————————————————————

            
            if idx - last_idx >= frame_gap_threshold:
                final_frames.append(idx)
                last_idx = idx
            else:
                changed = detect_scene_change_gt(seq_name, last_idx, idx, step3_dir, iou_thresh_gt)
                if not changed:
                    changed = detect_scene_change_soi(seq_name, last_idx, idx, step3_dir, iou_thresh_soi)
                if changed:
                    final_frames.append(idx)
                    last_idx = idx

        save_path = os.path.join(save_dir, f"{seq_name}_soi_frames.jsonl")
        with open(save_path, 'w') as f:
            json.dump(final_frames, f, ensure_ascii=False)
        print(f"✅ {seq_name}: 初筛 {len(soi_candidates)} ➜ 标注 {len(final_frames)} ➜ 写入 {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lasot")
    parser.add_argument("--status_dir", type=str, default="/home/wyp/project/SUTrack/soi/trackers_status")
    parser.add_argument("--step3_dir", type=str, default="/home/wyp/project/SUTrack/soi/step3_1_results")
    parser.add_argument("--save_dir", type=str, default="/home/wyp/project/SUTrack/soi/step3_2_results1111111111111")
    parser.add_argument("--frame_gap_threshold", type=int, default=10)
    parser.add_argument("--iou_thresh_gt", type=float, default=0.5)
    parser.add_argument("--iou_thresh_soi", type=float, default=0.4)
    parser.add_argument("--min_vote_ratio", type=float, default=0.5)
    args = parser.parse_args()

    extract_soi_frames(
        args.status_dir,
        args.step3_dir,
        args.save_dir,
        args.dataset_name,
        args.frame_gap_threshold,
        args.iou_thresh_gt,
        args.iou_thresh_soi,
        args.min_vote_ratio,
    )
