# pytracking/soi_pipeline/core/data_processor.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from .box_utils import BoundingBox, is_visually_similar, non_maximum_suppression


class DataProcessor:
    """数据处理器 - 负责Step 3.1的框过滤和合并"""
    
    def __init__(self, config):
        self.config = config
    
    def load_jsonl(self, path: str) -> List[Dict]:
        """加载JSONL文件"""
        if not os.path.exists(path):
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            return []
    
    def collect_soi_boxes(self, seq_name: str, frame_idx: int) -> List[BoundingBox]:
        """收集SOI跟踪器的预测框"""
        soi_boxes = []
        
        if not os.path.exists(self.config.soi_tracker_dir):
            return soi_boxes
        
        for tracker_name in os.listdir(self.config.soi_tracker_dir):
            tracker_dir = os.path.join(self.config.soi_tracker_dir, tracker_name)
            if not os.path.isdir(tracker_dir):
                continue
            
            soi_file = os.path.join(tracker_dir, f"{seq_name}_mask.jsonl")
            soi_data = self.load_jsonl(soi_file)
            
            if frame_idx < len(soi_data):
                for box_dict in soi_data[frame_idx]:
                    try:
                        box = BoundingBox(
                            box_dict['x1'], box_dict['y1'], 
                            box_dict['x2'], box_dict['y2']
                        )
                        if box.is_valid():
                            soi_boxes.append(box)
                    except (KeyError, TypeError):
                        continue
        
        return soi_boxes
    
    def collect_detection_boxes(self, seq_name: str, frame_idx: int) -> List[BoundingBox]:
        """收集检测器的预测框"""
        det_boxes = []
        
        det_file = os.path.join(self.config.detection_dir, f"{seq_name}.jsonl")
        det_data = self.load_jsonl(det_file)
        
        if frame_idx < len(det_data):
            for box_dict in det_data[frame_idx]:
                try:
                    box = BoundingBox(
                        box_dict['x1'], box_dict['y1'],
                        box_dict['x2'], box_dict['y2']
                    )
                    if box.is_valid():
                        det_boxes.append(box)
                except (KeyError, TypeError):
                    continue
        
        return det_boxes
    
    def merge_and_filter_frame(self, gt_box: BoundingBox, 
                              soi_boxes: List[BoundingBox], 
                              det_boxes: List[BoundingBox]) -> List[BoundingBox]:
        """合并和过滤单帧的所有框"""
        all_boxes = soi_boxes + det_boxes
        
        if not all_boxes:
            return [gt_box]
        
        # 过滤与GT太相似的框
        filtered_boxes = []
        for box in all_boxes:
            if not is_visually_similar(
                box, gt_box, 
                iou_thresh=self.config.iou_thresh_gt,
                center_thresh=self.config.center_thresh,
                scale_thresh=self.config.scale_thresh
            ):
                filtered_boxes.append(box)
        
        # 应用NMS去除重复框
        final_boxes = non_maximum_suppression(
            filtered_boxes, 
            iou_thresh=self.config.iou_thresh_nms
        )
        
        return [gt_box] + final_boxes
    
    # def process_sequence(self, seq_name: str, ground_truth_rects: List[List[float]]) -> List[List[List[float]]]:
    #     """处理整个序列的所有帧"""
    #     results = []
        
    #     for frame_idx, gt_rect in enumerate(ground_truth_rects):
    #         # 转换GT格式
    #         gt_box = BoundingBox.from_xywh(gt_rect)
            
    #         # 收集各种预测框
    #         soi_boxes = self.collect_soi_boxes(seq_name, frame_idx)
    #         det_boxes = self.collect_detection_boxes(seq_name, frame_idx)
            
    #         # 合并和过滤
    #         merged_boxes = self.merge_and_filter_frame(gt_box, soi_boxes, det_boxes)
            
    #         # 转换为列表格式保存
    #         frame_result = [box.to_list() for box in merged_boxes]
    #         results.append(frame_result)
        
    #     return results
    
    def process_sequence(self, seq_name: str, dataset_name: str, ground_truth_rects: List[List[float]]) -> List[List[List[float]]]:
        # 预加载所有 soi/det 数据
        soi_data_all = {}
        for tracker_name in os.listdir(self.config.soi_tracker_dir):
            tracker_dir = os.path.join(self.config.soi_tracker_dir, tracker_name)
            if not os.path.isdir(tracker_dir):
                continue
            if not os.path.exists(os.path.join(tracker_dir, f"{seq_name}_mask.jsonl")):
                soi_file = os.path.join(tracker_dir, dataset_name, f"{seq_name}_mask.jsonl")
            else:
                soi_file = os.path.join(tracker_dir, f"{seq_name}_mask.jsonl")
            soi_data_all[tracker_name] = self.load_jsonl(soi_file)

        if self.config.detection_dir:
            det_data = self.load_jsonl(os.path.join(self.config.detection_dir, f"{seq_name}.jsonl"))
        else:
            det_data = None
        
        results = []
        for frame_idx, gt_rect in enumerate(ground_truth_rects):
            gt_box = BoundingBox.from_xywh(gt_rect)

            # 所有soi tracker一并收集
            soi_boxes = []
            for tracker_name, soi_data in soi_data_all.items():
                if frame_idx < len(soi_data):
                    for box_dict in soi_data[frame_idx]:
                        try:
                            box = BoundingBox(
                                box_dict['x1'], box_dict['y1'], 
                                box_dict['x2'], box_dict['y2']
                            )
                            if box.is_valid():
                                soi_boxes.append(box)
                        except (KeyError, TypeError):
                            continue
            det_boxes = []
            if det_data:
                if frame_idx < len(det_data):
                    for box_dict in det_data[frame_idx]:
                        try:
                            box = BoundingBox(
                                box_dict['x1'], box_dict['y1'],
                                box_dict['x2'], box_dict['y2']
                            )
                            if box.is_valid():
                                det_boxes.append(box)
                        except (KeyError, TypeError):
                            continue

            # 后续同原逻辑
            merged_boxes = self.merge_and_filter_frame(gt_box, soi_boxes, det_boxes)
            frame_result = [box.to_list() for box in merged_boxes]
            results.append(frame_result)

        return results
