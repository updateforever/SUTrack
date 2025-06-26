# pytracking/soi_pipeline/core/box_utils.py
import math
import numpy as np
from typing import List, Tuple, Union


class BoundingBox:
    """边界框工具类"""
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return max(0, self.width * self.height)
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-6)
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_xywh(self) -> List[float]:
        return [self.x1, self.y1, self.width, self.height]
    
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0
    
    @classmethod
    def from_list(cls, coords: List[float]) -> 'BoundingBox':
        return cls(coords[0], coords[1], coords[2], coords[3])
    
    @classmethod
    def from_xywh(cls, xywh: List[float]) -> 'BoundingBox':
        x, y, w, h = xywh
        return cls(x, y, x + w, y + h)


def compute_iou(box1: BoundingBox, box2: BoundingBox, iou_type: str = "ciou") -> float:
    """计算两个框的IoU/DIoU/CIoU"""
    # 交集计算
    x_left = max(box1.x1, box2.x1)
    y_top = max(box1.y1, box2.y1)
    x_right = min(box1.x2, box2.x2)
    y_bottom = min(box1.y2, box2.y2)
    
    inter_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    union_area = box1.area + box2.area - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    if iou_type == "iou":
        return iou
    
    # 中心点距离
    cx1, cy1 = box1.center
    cx2, cy2 = box2.center
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    # 最小包围框对角线
    enclose_x1 = min(box1.x1, box2.x1)
    enclose_y1 = min(box1.y1, box2.y1)
    enclose_x2 = max(box1.x2, box2.x2)
    enclose_y2 = max(box1.y2, box2.y2)
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    diou = iou - center_dist_sq / (enclose_diag_sq + 1e-6)
    
    if iou_type == "diou":
        return diou
    
    # CIoU的v参数
    v = (4 / math.pi ** 2) * (
        math.atan(box1.aspect_ratio) - math.atan(box2.aspect_ratio)
    ) ** 2
    alpha = v / (1 - iou + v + 1e-6)
    
    return diou - alpha * v


def is_visually_similar(box1: BoundingBox, box2: BoundingBox, 
                       iou_thresh: float = 0.4, center_thresh: float = 20.0, 
                       scale_thresh: float = 0.3) -> bool:
    """判断两个框是否视觉相似"""
    # IoU检查
    iou = compute_iou(box1, box2, iou_type="ciou")
    if iou > iou_thresh:
        return True
    
    # 中心距离检查
    cx1, cy1 = box1.center
    cx2, cy2 = box2.center
    center_dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    if center_dist < center_thresh:
        return True
    
    # 尺度差异检查
    w1, h1 = box1.width, box1.height
    w2, h2 = box2.width, box2.height
    scale_diff = max(
        abs(w1 - w2) / max(w1, w2, 1e-6),
        abs(h1 - h2) / max(h1, h2, 1e-6)
    )
    
    return scale_diff < scale_thresh


def non_maximum_suppression(boxes: List[BoundingBox], iou_thresh: float = 0.4) -> List[BoundingBox]:
    """非极大值抑制"""
    if not boxes:
        return []
    
    keep = []
    for box in boxes:
        is_suppressed = any(
            is_visually_similar(box, kept_box, iou_thresh=iou_thresh) 
            for kept_box in keep
        )
        if not is_suppressed:
            keep.append(box)
    
    return keep


def is_small_box(box: BoundingBox, img_width: int, img_height: int,
                area_thresh: float = 0.00005, min_size: int = 16) -> bool:
    """判断是否为小目标框"""
    if not box.is_valid():
        return True
    
    area_ratio = box.area / (img_width * img_height)
    return (area_ratio < area_thresh or 
            box.width < min_size or 
            box.height < min_size)