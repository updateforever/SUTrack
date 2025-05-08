# step3_merge_filter.py

import os
import sys
import json
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.ops import nms
from lib.test.evaluation import get_dataset
from typing import List
import math

def box_visually_similar(box1: List[float], box2: List[float], 
                         iou_thresh=0.4, center_thresh=20, scale_thresh=0.3, iou_type="ciou") -> bool:
    """
    判断两个框在视觉上是否“过于相似”，用于去冗余判断。
    - iou_type: 可选 'iou'（默认）/ 'diou' / 'ciou'
    - iou_thresh：重叠度
    - center_thresh：中心点距离阈值（像素）
    - scale_thresh：尺寸差异比例
    """

    # IoU 判断（可选 diou / ciou）
    iou = box_iou_single(box1, box2, iou_type=iou_type)
    if iou > iou_thresh:
        return True

    # 中心点距离
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    center_dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    if center_dist < center_thresh:
        return True

    # 尺寸差异（长宽比例）
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    scale_diff = max(abs(w1 - w2) / max(w1, w2 + 1e-6), abs(h1 - h2) / max(h1, h2 + 1e-6))
    if scale_diff < scale_thresh:
        return True

    return False


def box_iou_single(box1, box2, iou_type="iou"):
    """
    支持 IoU / DIoU / CIoU
    box: [x1, y1, x2, y2]
    """
    # 交集面积
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # 面积
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)

    if iou_type == "iou":
        return iou

    # 中心点距离平方
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # 外接框对角线平方
    xC1 = min(box1[0], box2[0])
    yC1 = min(box1[1], box2[1])
    xC2 = max(box1[2], box2[2])
    yC2 = max(box1[3], box2[3])
    enclose_diag_sq = (xC2 - xC1) ** 2 + (yC2 - yC1) ** 2

    diou = iou - center_dist_sq / (enclose_diag_sq + 1e-6)

    if iou_type == "diou":
        return diou

    # aspect ratio penalty (CIoU)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    v = (4 / (math.pi ** 2)) * (math.atan(w1 / h1 + 1e-6) - math.atan(w2 / h2 + 1e-6)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)

    return diou - alpha * v  # CIoU

def box_xyxy_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f]


def merge_and_filter_one_frame(gt_box, soi_boxes_all, det_boxes,
                               iou_thresh_gt=0.7, iou_thresh_nms=0.5):
    """
    合并所有候选框并去重，保留干净SOI集合。
    输出：[GT, SOI1, SOI2, ...]，所有为 int(x1,y1,x2,y2)
    """
    all_boxes = soi_boxes_all + det_boxes
    if len(all_boxes) == 0:
        return [gt_box]

    # Step 1: 去掉与 GT 重合过高的框（预测正确）
    filtered_boxes = []
    for box in all_boxes:
        if box_iou_single(box, gt_box) < iou_thresh_gt:
            filtered_boxes.append(box)

    # Step 2: 去除候选之间重复框（手动 NMS）
    final_boxes = remove_high_iou_boxes(filtered_boxes, iou_thresh=iou_thresh_nms)

    # 输出结构：第一框为GT
    return [list(map(int, gt_box))] + [list(map(int, b)) for b in final_boxes]

def remove_high_iou_boxes(boxes, iou_thresh=0.3):
    """
    视觉相似性去冗余：IoU + 中心点 + 尺寸
    """
    keep = []
    for i, box in enumerate(boxes):
        should_keep = True
        for kept in keep:
            if box_visually_similar(box, kept, iou_thresh=iou_thresh):
                should_keep = False
                break
        if should_keep:
            keep.append(box)
    return keep


# def remove_high_iou_boxes(boxes, iou_thresh=0.5):
#     """
#     手动实现无置信度的 IoU 去冗余
#     输入: boxes: List[List[int]]
#     输出: 保留后的 boxes
#     """
#     keep = []
#     for i, box in enumerate(boxes):
#         should_keep = True
#         for kept in keep:
#             if box_iou_single(box, kept) > iou_thresh:
#                 should_keep = False
#                 break
#         if should_keep:
#             keep.append(box)
#     return keep


# def box_iou_single(box1, box2):
#     """
#     计算两个 box（x1y1x2y2）之间的 IoU
#     """
#     xA = max(box1[0], box2[0])
#     yA = max(box1[1], box2[1])
#     xB = min(box1[2], box2[2])
#     yB = min(box1[3], box2[3])

#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
#     boxBArea = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))

#     return interArea / (boxAArea + boxBArea - interArea + 1e-6)



def process_dataset(
    dataset_name: str,
    soi_dir: str,
    det_dir: str,
    save_dir: str = "./step3_filtered",
    iou_thresh_gt: float = 0.7,
    nms_thresh: float = 0.5
):
    os.makedirs(save_dir, exist_ok=True)
    dataset = get_dataset(dataset_name)

    for seq in tqdm(dataset, desc="Step 3 Filtering"):
        gt_boxes = seq.ground_truth_rect
        seq_name = seq.name
        if seq_name != "swing-17":
            continue
        det_jsonl = os.path.join(det_dir, f"{seq_name}.jsonl")
        if not os.path.exists(det_jsonl):
            print(f"⚠️ 缺少检测结果: {seq_name}")
            continue
        det_data = load_jsonl(det_jsonl)

        soi_datas = []
        for tracker_name in os.listdir(soi_dir):
            tracker_dir = os.path.join(soi_dir, tracker_name)
            if not os.path.isdir(tracker_dir):
                continue
            soi_path = os.path.join(tracker_dir, f"{seq_name}_mask.jsonl")
            if os.path.isfile(soi_path):
                soi_datas.append(load_jsonl(soi_path))

        output_path = os.path.join(save_dir, f"{seq_name}.jsonl")
        with open(output_path, "w") as f_out:
            for idx in range(len(gt_boxes)):
                if idx == 25:
                    print("skip")
                gt_xywh = gt_boxes[idx]
                gt_box = [gt_xywh[0], gt_xywh[1], gt_xywh[0] + gt_xywh[2], gt_xywh[1] + gt_xywh[3]]

                soi_boxes_all = []
                for soi_data in soi_datas:
                    if idx < len(soi_data):
                        soi_boxes_all += [
                            [b["x1"], b["y1"], b["x2"], b["y2"]] for b in soi_data[idx]
                        ]

                det_boxes = []
                if idx < len(det_data):
                    det_boxes = [
                        [b["x1"], b["y1"], b["x2"], b["y2"]] for b in det_data[idx]
                    ]

                merged = merge_and_filter_one_frame(gt_box, soi_boxes_all, det_boxes,
                                                    iou_thresh_gt=iou_thresh_gt, iou_thresh_nms=nms_thresh)
                f_out.write(json.dumps(merged, ensure_ascii=False) + "\n")

        print(f"✅ {seq_name} 已处理完成，输出: {output_path}")


def load_candidate_boxes(step3_dir: str, seq_name: str, frame_idx: int) -> List[List[float]]:
    """
    读取指定序列和帧的候选框（第一个为 GT 框）
    """
    file_path = os.path.join(step3_dir, f"{seq_name}.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到候选框文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if i == frame_idx:
                data = json.loads(line.strip())
                return data.get("boxes", [])
    return []

def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def detect_scene_change_gt(prev_boxes: List[List[float]], curr_boxes: List[List[float]], iou_thresh: float = 0.7) -> bool:
    """判断 GT 框是否剧烈变化（即 IOU 低于阈值）"""
    if not prev_boxes or not curr_boxes:
        return True
    iou = compute_iou(prev_boxes[0], curr_boxes[0])
    return iou < iou_thresh

def detect_scene_change_soi(prev_boxes: List[List[float]], curr_boxes: List[List[float]], iou_thresh: float = 0.5, min_matched: int = 1) -> bool:
    """根据候选框匹配数判断是否发生剧烈变化"""
    if not prev_boxes or not curr_boxes:
        return True

    matched = 0
    for pb in prev_boxes[1:]:  # 跳过GT
        for cb in curr_boxes[1:]:
            if compute_iou(pb, cb) > iou_thresh:
                matched += 1
                break
    return matched < min_matched


# ✅ CLI 入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: 合并候选框并清洗")

    parser.add_argument('--dataset_name', type=str, default='lasot', help='数据集名称')
    parser.add_argument('--soi_dir', type=str, default='/home/wyp/project/SUTrack/soi/tracker_soi_results', help='跟踪器SOI框文件夹')
    parser.add_argument('--det_dir', type=str, default='/home/wyp/project/ultralytics/yoloworld_results/', help='检测框文件夹（YOLO等）')
    parser.add_argument('--save_dir', type=str, default='/home/wyp/project/SUTrack/soi/step3_1_results', help='保存路径')
    parser.add_argument('--iou_thresh_gt', type=float, default=0.5, help='GT去重阈值')
    parser.add_argument('--nms_thresh', type=float, default=0.4, help='NMS阈值')

    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset_name,
        soi_dir=args.soi_dir,
        det_dir=args.det_dir,
        save_dir=args.save_dir,
        iou_thresh_gt=args.iou_thresh_gt,
        nms_thresh=args.nms_thresh
    )
