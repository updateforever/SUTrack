# pytracking/soi_pipeline/core/frame_extractor.py
import os
import json
import math
import cv2
from typing import List, Dict
from .box_utils import BoundingBox, compute_iou, is_small_box


class FrameExtractor:
    """SOI帧提取器 - 负责Step 3.2的关键帧提取"""
    
    def __init__(self, config):
        self.config = config
    
    def load_tracker_status(self, seq_name: str, dataset_name: str) -> List[List[str]]:
        """加载所有跟踪器的状态文件"""
        trackers = []
        
        if not os.path.exists(self.config.status_dir):
            return trackers
        
        for tracker_name in os.listdir(self.config.status_dir):
            tracker_dir = os.path.join(self.config.status_dir, tracker_name)

            if not os.path.exists(os.path.join(tracker_dir, f"{seq_name}_mask.jsonl")):
                status_file = os.path.join(tracker_dir, dataset_name, f"{seq_name}_status.txt")
            else:
                status_file = os.path.join(tracker_dir, f"{seq_name}_status.txt")
            
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines()]
                        if lines:
                            trackers.append(lines)
                except Exception as e:
                    print(f"Warning: Failed to load status for {tracker_name}: {e}")
        
        return trackers
    
    def load_candidate_boxes(self, jsonl_path: str, frame_idx: int) -> List[BoundingBox]:
        """从step3.1的结果中加载候选框"""
        if not os.path.exists(jsonl_path):
            return []
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if frame_idx >= len(lines):
                return []
            
            box_data = json.loads(lines[frame_idx].strip())
            return [BoundingBox.from_list(box_coords) for box_coords in box_data]
        
        except Exception as e:
            print(f"Warning: Failed to load boxes for frame {frame_idx}: {e}")
            return []
    
    def extract_soi_candidates(self, trackers: List[List[str]], 
                              seq_len: int, ground_truth_rects: List[List[float]]) -> List[int]:
        """基于跟踪器状态投票提取SOI候选帧"""
        candidates = []
        
        for frame_idx in range(seq_len - 1):
            # 收集所有跟踪器在该帧的投票
            votes = []
            for tracker_status in trackers:
                if frame_idx < len(tracker_status):
                    votes.append(tracker_status[frame_idx])
            
            if not votes:
                continue
            
            # 跳过所有跟踪器都标记为absent的帧
            if all(vote == 'absent' for vote in votes):
                continue
            
            # 跳过GT为全零的帧
            if sum(ground_truth_rects[frame_idx]) == 0:
                continue
            
            # 计算失败/漂移的比例
            fail_votes = sum(1 for vote in votes if vote in ('Drift', 'Fail'))
            fail_ratio = fail_votes / len(votes)
            
            # 超过阈值则认为是SOI帧
            if fail_ratio >= self.config.min_vote_ratio:
                candidates.append(frame_idx + 1)  # 使用1-based索引
        
        return candidates
    
    def detect_scene_change(self, boxes1: List[BoundingBox], boxes2: List[BoundingBox]) -> bool:
        """检测两帧之间是否发生场景变化（更宽松）"""
        if not boxes1 or not boxes2:
            return True  # 缺失时仍视为变化

        gt1, gt2 = boxes1[0], boxes2[0]
        gt_iou = compute_iou(gt1, gt2, iou_type="iou")

        # 🎯 更宽松：GT变化阈值从 0.15 → 0.05（除非非常剧烈，不判定为变化）
        if gt_iou < 0.1:
            return True

        # 🎯 宽高比变化阈值从 0.8 → 1.2（对形变更容忍）
        ratio1 = gt1.aspect_ratio
        ratio2 = gt2.aspect_ratio
        ratio_change = abs(math.log(ratio1 + 1e-6) - math.log(ratio2 + 1e-6))
        if ratio_change > 1.2:
            return True

        # 🎯 中心位移：放宽归一化位移阈值 0.4 → 0.8
        cx1, cy1 = gt1.center
        cx2, cy2 = gt2.center
        diag = math.sqrt(gt1.width**2 + gt1.height**2)
        center_dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        normalized_dist = center_dist / (diag + 1e-6)
        if normalized_dist > 0.8:
            return True

        # 🎯 SOI匹配比例判断
        if len(boxes1) > 1 and len(boxes2) > 1:
            soi_boxes1 = boxes1[1:]
            soi_boxes2 = boxes2[1:]

            matched = 0
            for box1 in soi_boxes1:
                for box2 in soi_boxes2:
                    if compute_iou(box1, box2, iou_type="iou") > 0.25:
                        matched += 1
                        break

            matched_ratio = matched / max(len(soi_boxes1), 1)
            if matched_ratio < getattr(self.config, 'min_soi_match_ratio', 0.2):
                return True

        return False

    
    def extract_soi_frames(self, seq_name: str, dataset_name: str, frames: List[str], 
                          ground_truth_rects: List[List[float]], 
                          step3_1_jsonl: str) -> List[int]:
        """提取最终的SOI帧列表"""
        # 加载跟踪器状态
        trackers = self.load_tracker_status(seq_name, dataset_name)
        if not trackers:
            if self.config.verbose:
                print(f"Warning: No tracker status found for {seq_name}")
            return []
        
        # 阶段1：基于投票提取候选帧
        candidates = self.extract_soi_candidates(trackers, len(frames), ground_truth_rects)
        if not candidates:
            return []
        
        # 阶段2：基于场景变化和质量过滤
        final_frames = []
        last_selected_idx = -self.config.frame_gap_threshold - 1
        
        for frame_idx in candidates:
            # 检查候选框是否足够（至少GT + 1个干扰框）
            try:
                boxes = self.load_candidate_boxes(step3_1_jsonl, frame_idx)
                if len(boxes) <= 1:
                    continue
            except Exception:
                continue
            
            # 检查GT是否过小
            try:
                if frame_idx < len(frames):
                    img = cv2.imread(frames[frame_idx])
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                        gt_box = boxes[0]
                        if is_small_box(gt_box, img_w, img_h, 
                                       area_thresh=self.config.small_area_thresh,
                                       min_size=self.config.min_box_size):
                            if self.config.verbose:
                                print(f"Skipping {seq_name} frame {frame_idx}: GT too small")
                            continue
            except Exception:
                continue
            
            # 应用帧间隔阈值
            if frame_idx - last_selected_idx >= self.config.frame_gap_threshold:
                final_frames.append(frame_idx)
                last_selected_idx = frame_idx
            else:
                # 检查是否发生场景变化
                try:
                    last_boxes = self.load_candidate_boxes(step3_1_jsonl, last_selected_idx)
                    curr_boxes = self.load_candidate_boxes(step3_1_jsonl, frame_idx)
                    
                    if self.detect_scene_change(last_boxes, curr_boxes):
                        final_frames.append(frame_idx)
                        last_selected_idx = frame_idx
                except Exception:
                    continue
        
        return final_frames