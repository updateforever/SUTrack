import os
import sys
import json

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import math
import torch
import numpy as np
from tqdm import tqdm
from lib.test.evaluation import get_dataset
from typing import List

def is_valid_box(box):
    return (box[2] > box[0]) and (box[3] > box[1])

def box_iou_single(box1, box2, iou_type="ciou"):
    # 交集面积
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)

    if iou_type == "iou":
        return iou

    # DIoU & CIoU components
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    center_dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2

    xC1 = min(box1[0], box2[0])
    yC1 = min(box1[1], box2[1])
    xC2 = max(box1[2], box2[2])
    yC2 = max(box1[3], box2[3])
    enclose_diag_sq = (xC2 - xC1)**2 + (yC2 - yC1)**2

    diou = iou - center_dist_sq / (enclose_diag_sq + 1e-6)
    if iou_type == "diou":
        return diou

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    v = (4 / math.pi**2) * (math.atan(w1 / h1 + 1e-6) - math.atan(w2 / h2 + 1e-6)) ** 2
    alpha = v / (1 - iou + v + 1e-6)

    return diou - alpha * v


def box_visually_similar(box1: List[float], box2: List[float],
                         iou_thresh=0.4, center_thresh=20, scale_thresh=0.3, iou_type="ciou") -> bool:
    iou = box_iou_single(box1, box2, iou_type=iou_type)
    if iou > iou_thresh:
        return True

    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    if ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5 < center_thresh:
        return True

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    scale_diff = max(abs(w1 - w2) / max(w1, w2 + 1e-6), abs(h1 - h2) / max(h1, h2 + 1e-6))
    return scale_diff < scale_thresh


def remove_high_iou_boxes(boxes, iou_thresh=0.4):
    keep = []
    for box in boxes:
        if all(not box_visually_similar(box, k, iou_thresh=iou_thresh) for k in keep):
            keep.append(box)
    return keep


def merge_and_filter_one_frame(gt_box, soi_boxes_all, det_boxes,
                               iou_thresh_gt=0.5, iou_thresh_nms=0.4):
    all_boxes = soi_boxes_all + det_boxes  # 多个预测框可能与gt重叠，但并没有带gt值
    if len(all_boxes) == 0:
        return [gt_box]

    filtered_boxes = [b for b in all_boxes if not box_visually_similar(b, gt_box, iou_thresh=iou_thresh_gt)]
    final_boxes = remove_high_iou_boxes(filtered_boxes, iou_thresh=iou_thresh_nms)
    return [list(map(int, gt_box))] + [list(map(int, b)) for b in final_boxes]


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f]


def process_dataset(dataset_name, soi_dir, det_dir, save_dir, iou_thresh_gt=0.5, nms_thresh=0.4):
    os.makedirs(save_dir, exist_ok=True)
    dataset = get_dataset(dataset_name)

    for seq in tqdm(dataset, desc="Step 3 Filtering"):
        gt_boxes = seq.ground_truth_rect
        seq_name = seq.name

        # if seq_name != "swing-17":
        #     continue
        det_path = os.path.join(det_dir, f"{seq_name}.jsonl")
        if not os.path.exists(det_path):
            print(f"⚠️ 缺少检测结果: {seq_name}")
            continue
        det_data = load_jsonl(det_path)

        soi_datas = []
        for tracker_name in os.listdir(soi_dir):
            path = os.path.join(soi_dir, tracker_name, f"{seq_name}_mask.jsonl")
            if os.path.isfile(path):
                soi_datas.append(load_jsonl(path))

        save_path = os.path.join(save_dir, f"{seq_name}.jsonl")
        with open(save_path, "w") as f_out:
            for idx in range(len(gt_boxes)):
                # if idx== 25:
                #     print("skip")
                gt = gt_boxes[idx]
                gt_box = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]

                soi_boxes = []
                for soi in soi_datas:
                    if idx < len(soi):
                        soi_boxes += [[b['x1'], b['y1'], b['x2'], b['y2']] for b in soi[idx]]
                        soi_boxes = [b for b in soi_boxes if is_valid_box(b)]

                det_boxes = []
                if idx < len(det_data):
                    det_boxes = [[b['x1'], b['y1'], b['x2'], b['y2']] for b in det_data[idx]]
                    det_boxes = [b for b in det_boxes if is_valid_box(b)]

                merged = merge_and_filter_one_frame(gt_box, soi_boxes, det_boxes,
                                                    iou_thresh_gt=iou_thresh_gt,
                                                    iou_thresh_nms=nms_thresh)
                f_out.write(json.dumps(merged, ensure_ascii=False) + "\n")

        print(f"✅ {seq_name} 已处理完成，输出: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='lasot', help='数据集名称')
    parser.add_argument('--soi_dir', type=str, default='/home/wyp/project/SUTrack/soi/tracker_soi_results', help='跟踪器SOI框文件夹')
    parser.add_argument('--det_dir', type=str, default='/home/wyp/project/ultralytics/yoloworld_results/', help='检测框文件夹（YOLO等）')
    parser.add_argument('--save_dir', type=str, default='/home/wyp/project/SUTrack/soi/step3_1_results111111', help='保存路径')
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
