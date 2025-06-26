# pytracking/soi_pipeline/pipelines/step6_pipeline_enhanced.py
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from collections import defaultdict

from ..utils.verification_utils import load_jsonl, save_jsonl, compute_iou, xywh_to_xyxy

# 在step6_pipeline.py中
from ..configs.config import HumanExperimentConfig, Config, BatchConfig


def filter_vlm_failed_samples(step5_dir: str, config: HumanExperimentConfig) -> List[Dict]:
    """
    筛选VLM失败的样本
    
    筛选逻辑：
    1. IoU_1234 < 阈值
    2. 语义层级越丰富效果越差（level degradation）
    3. 确保样本分布合理
    """
    print("🔍 Filtering VLM failed samples...")
    
    failed_samples = []
    
    # 遍历所有验证结果文件
    for filename in os.listdir(step5_dir):
        if not filename.endswith("_verify.jsonl"):
            continue
            
        seq_name = filename.replace("_verify.jsonl", "")
        file_path = os.path.join(step5_dir, filename)
        
        try:
            records = load_jsonl(file_path)
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            continue
        
        seq_failed_samples = []
        
        for record in records:
            # 检查是否为VLM失败样本
            is_failed = False
            failure_reasons = []
            
            # 条件1: IoU过低
            iou_1234 = record.get("iou_1234", 0)
            if iou_1234 < config.vlm_failure_threshold:
                is_failed = True
                failure_reasons.append(f"low_iou_{iou_1234:.3f}")
            
            # 条件2: 语义层级退化（越详细效果越差）
            level_ious = []
            for level in ["12", "123", "1234"]:
                iou = record.get(f"iou_{level}", 0)  # iou_
                level_ious.append(iou)
            
            # 检查是否存在退化趋势
            if len(level_ious) >= 2:
                degradation_count = 0
                for i in range(1, len(level_ious)):
                    if level_ious[i] < level_ious[i-1]:
                        degradation_count += 1
                
                # 如果大部分层级都在退化
                if degradation_count >= len(level_ious) - 1:
                    is_failed = True
                    failure_reasons.append("semantic_degradation")
            
            # 条件3: 所有层级都失败
            all_failed = all(not record.get(f"ok_{level}", False) 
                           for level in ["12", "123", "1234"])
            if all_failed:
                is_failed = True
                failure_reasons.append("all_levels_failed")
            
            if is_failed:
                record["failure_reasons"] = failure_reasons
                record["sequence_name"] = seq_name
                seq_failed_samples.append(record)
        
        # 控制每个序列的样本数量
        if seq_failed_samples:
            # 按IoU排序，优先选择中等难度的样本
            seq_failed_samples.sort(key=lambda x: x.get("iou_1234", 0))
            
            # 选择合适数量的样本
            num_samples = min(max(config.min_samples_per_sequence, len(seq_failed_samples)), 
                            config.max_samples_per_sequence)
            
            # 从中间部分选择（避免过于简单或过于困难）
            start_idx = max(0, (len(seq_failed_samples) - num_samples) // 2)
            selected_samples = seq_failed_samples[start_idx:start_idx + num_samples]
            
            failed_samples.extend(selected_samples)
            print(f"📋 Selected {len(selected_samples)} failed samples from {seq_name}")
    
    print(f"✅ Total VLM failed samples: {len(failed_samples)}")
    return failed_samples

def analyze_failure_patterns(failed_samples: List[Dict]) -> Dict:
    """分析VLM失败模式"""
    print("📊 Analyzing VLM failure patterns...")
    
    analysis = {
        "total_samples": len(failed_samples),
        "failure_reasons": {},
        "iou_distribution": [],
        "sequence_distribution": {},
        "level_performance": {}
    }
    
    # 统计失败原因
    for sample in failed_samples:
        reasons = sample.get("failure_reasons", [])
        for reason in reasons:
            analysis["failure_reasons"][reason] = analysis["failure_reasons"].get(reason, 0) + 1
        
        # IoU分布
        analysis["iou_distribution"].append(sample.get("iou_1234", 0))
        
        # 序列分布
        seq_name = sample.get("sequence_name", "unknown")
        analysis["sequence_distribution"][seq_name] = analysis["sequence_distribution"].get(seq_name, 0) + 1
        
        # 各层级性能
        for level in ["12", "123", "1234"]:
            if level not in analysis["level_performance"]:
                analysis["level_performance"][level] = {"success": 0, "total": 0, "avg_iou": []}
            
            analysis["level_performance"][level]["total"] += 1
            if sample.get(f"ok_{level}", False):
                analysis["level_performance"][level]["success"] += 1
            
            iou = sample.get(f"iou_{level}", 0)
            analysis["level_performance"][level]["avg_iou"].append(iou)
    
    # 计算平均IoU
    for level in analysis["level_performance"]:
        ious = analysis["level_performance"][level]["avg_iou"]
        analysis["level_performance"][level]["avg_iou"] = np.mean(ious) if ious else 0
    
    return analysis

def create_balanced_batches(failed_samples: List[Dict], 
                          batch_size: int = 200,
                          stratify_by: str = "sequence_name",
                          min_batch_size: int = 50) -> List[List[Dict]]:
    """
    创建平衡的数据批次
    
    参数:
    - failed_samples: 失败样本列表
    - batch_size: 每批的目标大小
    - stratify_by: 分层依据 ("sequence_name", "iou_range", "failure_type")
    - min_batch_size: 最小批次大小
    
    返回:
    - 批次列表，每个批次是样本列表
    """
    print(f"📦 Creating balanced batches (target size: {batch_size})...")
    
    # 按指定策略分组
    if stratify_by == "sequence_name":
        groups = defaultdict(list)
        for sample in failed_samples:
            seq_name = sample.get("sequence_name", "unknown")
            groups[seq_name].append(sample)
    
    elif stratify_by == "iou_range":
        groups = defaultdict(list)
        for sample in failed_samples:
            iou = sample.get("iou_1234", 0)
            if iou < 0.1:
                range_key = "very_low_0.0-0.1"
            elif iou < 0.2:
                range_key = "low_0.1-0.2"
            elif iou < 0.3:
                range_key = "medium_0.2-0.3"
            elif iou < 0.4:
                range_key = "high_0.3-0.4"
            else:
                range_key = "very_high_0.4+"
            groups[range_key].append(sample)
    
    elif stratify_by == "failure_type":
        groups = defaultdict(list)
        for sample in failed_samples:
            reasons = sample.get("failure_reasons", [])
            primary_reason = reasons[0] if reasons else "unknown"
            if primary_reason.startswith("low_iou"):
                group_key = "low_iou"
            else:
                group_key = primary_reason
            groups[group_key].append(sample)
    
    else:
        # 默认：不分组，随机分批
        groups = {"all": failed_samples}
    
    print(f"📊 Found {len(groups)} groups: {list(groups.keys())}")
    for group_name, samples in groups.items():
        print(f"   - {group_name}: {len(samples)} samples")
    
    # 创建平衡批次
    batches = []
    group_names = list(groups.keys())
    group_indices = {name: 0 for name in group_names}
    
    while True:
        current_batch = []
        batch_full = False
        
        # 为当前批次从每个组中取样本
        for group_name in group_names:
            group_samples = groups[group_name]
            start_idx = group_indices[group_name]
            
            if start_idx >= len(group_samples):
                continue  # 该组已用完
            
            # 计算该组在当前批次中应占的样本数
            remaining_groups = sum(1 for gn in group_names 
                                 if group_indices[gn] < len(groups[gn]))
            if remaining_groups == 0:
                break
                
            samples_per_group = max(1, batch_size // remaining_groups)
            end_idx = min(start_idx + samples_per_group, len(group_samples))
            
            group_batch_samples = group_samples[start_idx:end_idx]
            current_batch.extend(group_batch_samples)
            group_indices[group_name] = end_idx
            
            if len(current_batch) >= batch_size:
                batch_full = True
                break
        
        if not current_batch:
            break  # 所有组都用完了
        
        # 如果批次太小且不是最后一批，尝试合并
        if len(current_batch) < min_batch_size:
            remaining_samples = []
            for group_name in group_names:
                group_samples = groups[group_name]
                start_idx = group_indices[group_name]
                remaining_samples.extend(group_samples[start_idx:])
            
            if remaining_samples:
                current_batch.extend(remaining_samples[:batch_size - len(current_batch)])
        
        if current_batch:
            # 打乱批次内的样本顺序
            random.shuffle(current_batch)
            batches.append(current_batch)
            print(f"📦 Created batch {len(batches)}: {len(current_batch)} samples")
        
        # 检查是否所有组都用完
        if all(group_indices[gn] >= len(groups[gn]) for gn in group_names):
            break
    
    print(f"✅ Created {len(batches)} batches")
    return batches

# def prepare_human_experiment_data_batched(failed_samples: List[Dict], 
#                                         dataset, 
#                                         output_dir: str,
#                                         batch_config: Dict = None) -> Dict[str, str]:
#     """
#     准备分批的人类实验数据
    
#     参数:
#     - failed_samples: 失败样本列表
#     - dataset: 数据集对象
#     - output_dir: 输出目录
#     - batch_config: 批次配置
#     """
#     print("📝 Preparing batched human experiment data...")
    
#     # 默认批次配置
#     if batch_config is None:
#         batch_config = {
#             "batch_size": 200,  # 每批200个样本
#             "stratify_by": "sequence_name",  # 按序列分层
#             "min_batch_size": 50,  # 最小批次大小
#             "shuffle_samples": True,  # 是否打乱样本
#             "create_overlap": False,  # 是否创建重叠批次用于验证
#             "overlap_ratio": 0.1  # 重叠比例
#         }
    
#     # 创建序列映射
#     seq_map = {seq.name: seq for seq in dataset}
    
#     # 如果需要打乱样本
#     if batch_config.shuffle_samples:  # 直接访问属性，不使用get()方法
#         random.shuffle(failed_samples)
    
#     # 创建批次
#     batches = create_balanced_batches(
#         failed_samples,
#         batch_size=batch_config.batch_size,  # 直接访问属性
#         stratify_by=batch_config.stratify_by,  # 直接访问属性
#         min_batch_size=batch_config.min_batch_size  # 直接访问属性
#     )
    
#     # 创建批次目录
#     batches_dir = os.path.join(output_dir, "experiment_batches")
#     os.makedirs(batches_dir, exist_ok=True)
    
#     batch_files = []
#     batch_metadata = []
    
#     for batch_idx, batch_samples in enumerate(batches):
#         print(f"📝 Processing batch {batch_idx + 1}/{len(batches)}...")
        
#         experiment_data = []
#         batch_stats = {
#             "batch_id": batch_idx + 1,
#             "total_samples": len(batch_samples),
#             "sequences": set(),
#             "iou_range": {"min": 1.0, "max": 0.0, "avg": 0.0},
#             "failure_types": defaultdict(int)
#         }
        
#         for i, sample in enumerate(batch_samples):
#             seq_name = sample.get("sequence_name")
#             frame_idx = sample.get("frame_idx")
            
#             if seq_name not in seq_map:
#                 continue
            
#             seq = seq_map[seq_name]
#             batch_stats["sequences"].add(seq_name)
            
#             # 统计信息
#             iou_1234 = sample.get("iou_1234", 0)
#             batch_stats["iou_range"]["min"] = min(batch_stats["iou_range"]["min"], iou_1234)
#             batch_stats["iou_range"]["max"] = max(batch_stats["iou_range"]["max"], iou_1234)
            
#             for reason in sample.get("failure_reasons", []):
#                 batch_stats["failure_types"][reason] += 1
            
#             # 找到参考帧（模板图）
#             template_frame_idx = 0  # 默认使用首帧
#             template_box = seq.ground_truth_rect[template_frame_idx].tolist()
            
#             # 准备实验条目
#             experiment_item = {
#                 "experiment_id": f"{seq_name}_{frame_idx:06d}",
#                 "batch_id": batch_idx + 1,
#                 "sample_id_in_batch": i + 1,
#                 "sequence_name": seq_name,
#                 "frame_idx": frame_idx,
#                 "template_frame_idx": template_frame_idx,
#                 "template_image_path": seq.frames[template_frame_idx],
#                 "current_image_path": seq.frames[frame_idx],
#                 "template_box": template_box,  # xywh format
#                 "ground_truth_box": sample.get("gt_box"),  # xywh format
                
#                 # VLM生成的描述
#                 "description_level_1": sample.get("vlm_output_cleaned", {}).get("level1", ""),
#                 "description_level_2": sample.get("vlm_output_cleaned", {}).get("level2", ""),
#                 "description_level_3": sample.get("vlm_output_cleaned", {}).get("level3", ""),
#                 "description_level_4": sample.get("vlm_output_cleaned", {}).get("level4", ""),
                
#                 # VLM验证结果（用于对比）
#                 "vlm_results": {
#                     "iou_12": sample.get("iou_12", 0),
#                     "iou_123": sample.get("iou_123", 0), 
#                     "iou_1234": sample.get("iou_1234", 0),
#                     "success_12": sample.get("ok_12", False),
#                     "success_123": sample.get("ok_123", False),
#                     "success_1234": sample.get("ok_1234", False)
#                 },
                
#                 # 失败信息
#                 "failure_reasons": sample.get("failure_reasons", []),
                
#                 # 待填入的人类实验结果
#                 "human_results": {
#                     "selected_level": None,  # 1, 2, 3, 4, 5 (无法确定)
#                     "bounding_box": None,    # [x, y, w, h]
#                     "confidence": None,      # 1-5
#                     "difficulty": None,      # 1-5
#                     "comments": "",
#                     "time_spent": 0         # 秒
#                 },
                
#                 "status": "pending"  # pending, completed, skipped
#             }
            
#             experiment_data.append(experiment_item)
        
#         # 计算统计信息
#         batch_stats["sequences"] = list(batch_stats["sequences"])
#         if batch_samples:
#             ious = [s.get("iou_1234", 0) for s in batch_samples]
#             batch_stats["iou_range"]["avg"] = np.mean(ious)
        
#         # 保存批次数据
#         batch_filename = f"batch_{batch_idx + 1:03d}_experiment_data.jsonl"
#         batch_file_path = os.path.join(batches_dir, batch_filename)
#         save_jsonl(batch_file_path, experiment_data)
#         batch_files.append(batch_file_path)
        
#         # 保存批次统计信息
#         stats_filename = f"batch_{batch_idx + 1:03d}_stats.json"
#         stats_file_path = os.path.join(batches_dir, stats_filename)
#         with open(stats_file_path, 'w', encoding='utf-8') as f:
#             json.dump(batch_stats, f, ensure_ascii=False, indent=2, default=str)
        
#         batch_metadata.append({
#             "batch_id": batch_idx + 1,
#             "filename": batch_filename,
#             "stats_file": stats_filename,
#             "sample_count": len(experiment_data),
#             "sequences": batch_stats["sequences"],
#             "estimated_time_minutes": len(experiment_data) * 0.5
#         })
        
#         print(f"   ✅ Batch {batch_idx + 1}: {len(experiment_data)} samples saved")
    
#     # 创建重叠批次（用于验证标注一致性）
#     if batch_config.create_overlap:
#         overlap_samples = create_overlap_batches(
#             batches, 
#             output_dir, 
#             overlap_ratio=batch_config.overlap_ratio
#         )
    
#     # 保存批次总览
#     batch_summary = {
#         "total_batches": len(batches),
#         "total_samples": len(failed_samples),
#         "batch_config": batch_config,
#         "created_date": pd.Timestamp.now().isoformat(),
#         "batches": batch_metadata
#     }
    
#     summary_file = os.path.join(batches_dir, "batches_summary.json")
    
#     with open(summary_file, 'w', encoding='utf-8') as f:
#         # 在保存之前转换 BatchConfig 对象为字典
#         if 'batch_config' in batch_summary and isinstance(batch_summary['batch_config'], BatchConfig):
#             batch_summary['batch_config'] = batch_summary['batch_config'].to_dict()

#         # 然后保存
#         json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    
#     # 生成批次分配建议
#     generate_batch_assignment_guide(batches_dir, batch_metadata)
    
#     print(f"✅ Created {len(batches)} experiment batches")
#     print(f"📁 Batches saved to: {batches_dir}")
    
#     return {
#         "batches_dir": batches_dir,
#         "batch_files": batch_files,
#         "summary_file": summary_file,
#         "total_batches": len(batches),
#         "total_samples": len(failed_samples)
#     }

def prepare_human_experiment_data_batched(failed_samples: List[Dict], 
                                        dataset, 
                                        output_dir: str,
                                        batch_config: BatchConfig = None) -> Dict[str, str]:
    """
    准备分批的人类实验数据
    
    参数:
    - failed_samples: 失败样本列表
    - dataset: 数据集对象
    - output_dir: 输出目录
    - batch_config: 批次配置
    """
    print("📝 Preparing batched human experiment data...")
    
    # 默认批次配置
    if batch_config is None:
        batch_config = BatchConfig()
    
    # 创建序列映射
    seq_map = {seq.name: seq for seq in dataset}
    
    # 如果需要打乱样本
    if batch_config.shuffle_samples:
        random.shuffle(failed_samples)
    
    # 创建批次
    batches = create_balanced_batches(
        failed_samples,
        batch_size=batch_config.batch_size,
        stratify_by=batch_config.stratify_by,
        min_batch_size=batch_config.min_batch_size
    )
    
    # 创建批次目录和图像目录
    batches_dir = os.path.join(output_dir, "experiment_batches")
    os.makedirs(batches_dir, exist_ok=True)
    
    # 创建图像目录
    images_dir = os.path.join(output_dir, "experiment_images")
    os.makedirs(images_dir, exist_ok=True)
    
    batch_files = []
    batch_metadata = []
    
    # 跟踪已复制的图像，避免重复复制
    copied_images = set()
    
    for batch_idx, batch_samples in enumerate(batches):
        print(f"📝 Processing batch {batch_idx + 1}/{len(batches)}...")
        
        # 为每个批次创建单独的图像目录
        batch_images_dir = os.path.join(images_dir, f"batch_{batch_idx + 1:03d}")
        os.makedirs(batch_images_dir, exist_ok=True)
        
        experiment_data = []
        batch_stats = {
            "batch_id": batch_idx + 1,
            "total_samples": len(batch_samples),
            "sequences": set(),
            "iou_range": {"min": 1.0, "max": 0.0, "avg": 0.0},
            "failure_types": defaultdict(int)
        }
        
        for i, sample in enumerate(batch_samples):
            seq_name = sample.get("sequence_name")
            frame_idx = sample.get("frame_idx")
            
            if seq_name not in seq_map:
                continue
            
            seq = seq_map[seq_name]
            batch_stats["sequences"].add(seq_name)
            
            # 统计信息
            iou_1234 = sample.get("iou_1234", 0)
            batch_stats["iou_range"]["min"] = min(batch_stats["iou_range"]["min"], iou_1234)
            batch_stats["iou_range"]["max"] = max(batch_stats["iou_range"]["max"], iou_1234)
            
            for reason in sample.get("failure_reasons", []):
                batch_stats["failure_types"][reason] += 1
            
            # 设置模板帧：使用第一帧和前一帧
            first_frame_idx = 0  # 第一帧
            prev_frame_idx = max(0, frame_idx - 1)  # 前一帧，确保不会是负数
            
            # 如果当前帧就是第一帧，则使用第一帧和第二帧
            if frame_idx == 0:
                prev_frame_idx = min(1, len(seq.frames) - 1)
            
            # 准备图像路径
            first_frame_path = seq.frames[first_frame_idx]
            prev_frame_path = seq.frames[prev_frame_idx]
            current_frame_path = seq.frames[frame_idx]
            
            # 复制图像到批次目录
            # 创建序列子目录
            seq_dir = os.path.join(batch_images_dir, seq_name)
            os.makedirs(seq_dir, exist_ok=True)
            
            # 目标文件名
            first_frame_dest = os.path.join(seq_dir, f"frame_{first_frame_idx:06d}.jpg")
            prev_frame_dest = os.path.join(seq_dir, f"frame_{prev_frame_idx:06d}.jpg")
            current_frame_dest = os.path.join(seq_dir, f"frame_{frame_idx:06d}.jpg")
            
            # 复制图像文件（如果尚未复制）
            image_paths = {
                "first_frame": (first_frame_path, first_frame_dest),
                "prev_frame": (prev_frame_path, prev_frame_dest),
                "current_frame": (current_frame_path, current_frame_dest)
            }
            
            for img_type, (src, dest) in image_paths.items():
                if dest not in copied_images:
                    try:
                        # 确保目标目录存在
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        # 复制图像
                        import shutil
                        shutil.copy2(src, dest)
                        copied_images.add(dest)
                    except Exception as e:
                        print(f"⚠️ Failed to copy {img_type} image: {e}")
                        # 如果复制失败，使用原始路径
                        if img_type == "first_frame":
                            first_frame_dest = first_frame_path
                        elif img_type == "prev_frame":
                            prev_frame_dest = prev_frame_path
                        elif img_type == "current_frame":
                            current_frame_dest = current_frame_path
            
            # 准备实验条目
            experiment_item = {
                "experiment_id": f"{seq_name}_{frame_idx:06d}",
                "batch_id": batch_idx + 1,
                "sample_id_in_batch": i + 1,
                "sequence_name": seq_name,
                "frame_idx": frame_idx,
                
                # 提供两个模板帧选择
                "first_frame_idx": first_frame_idx,
                "prev_frame_idx": prev_frame_idx,
                "first_frame_image_path": first_frame_dest,
                "prev_frame_image_path": prev_frame_dest,
                "current_image_path": current_frame_dest,
                
                # 两个模板帧的边界框
                "first_frame_box": seq.ground_truth_rect[first_frame_idx].tolist(),  # xywh format
                "prev_frame_box": seq.ground_truth_rect[prev_frame_idx].tolist(),  # xywh format
                
                "ground_truth_box": sample.get("gt_box"),  # xywh format
                
                # VLM生成的描述
                "description_level_1": sample.get("vlm_output_cleaned", {}).get("level1", ""),
                "description_level_2": sample.get("vlm_output_cleaned", {}).get("level2", ""),
                "description_level_3": sample.get("vlm_output_cleaned", {}).get("level3", ""),
                "description_level_4": sample.get("vlm_output_cleaned", {}).get("level4", ""),
                
                # VLM验证结果（用于对比）
                "vlm_results": {
                    "iou_12": sample.get("iou_12", 0),
                    "iou_123": sample.get("iou_123", 0), 
                    "iou_1234": sample.get("iou_1234", 0),
                    "success_12": sample.get("ok_12", False),
                    "success_123": sample.get("ok_123", False),
                    "success_1234": sample.get("ok_1234", False)
                },
                
                # 失败信息
                "failure_reasons": sample.get("failure_reasons", []),
                
                # 待填入的人类实验结果
                "human_results": {
                    "selected_level": None,  # 1, 2, 3, 4, 5 (无法确定)
                    "bounding_box": None,    # [x, y, w, h]
                    "confidence": None,      # 1-5
                    "difficulty": None,      # 1-5
                    "comments": "",
                    "time_spent": 0         # 秒
                },
                
                "status": "pending"  # pending, completed, skipped
            }
            
            experiment_data.append(experiment_item)
        
        # 计算统计信息
        batch_stats["sequences"] = list(batch_stats["sequences"])
        if batch_samples:
            ious = [s.get("iou_1234", 0) for s in batch_samples]
            batch_stats["iou_range"]["avg"] = np.mean(ious)
        
        # 保存批次数据
        batch_filename = f"batch_{batch_idx + 1:03d}_experiment_data.jsonl"
        batch_file_path = os.path.join(batches_dir, batch_filename)
        save_jsonl(batch_file_path, experiment_data)
        batch_files.append(batch_file_path)
        
        # 保存批次统计信息
        stats_filename = f"batch_{batch_idx + 1:03d}_stats.json"
        stats_file_path = os.path.join(batches_dir, stats_filename)
        with open(stats_file_path, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2, default=str)
        
        batch_metadata.append({
            "batch_id": batch_idx + 1,
            "filename": batch_filename,
            "images_dir": batch_images_dir,
            "stats_file": stats_filename,
            "sample_count": len(experiment_data),
            "sequences": batch_stats["sequences"],
            "estimated_time_minutes": len(experiment_data) * 0.5
        })
        
        print(f"   ✅ Batch {batch_idx + 1}: {len(experiment_data)} samples saved")
        print(f"   📸 Batch {batch_idx + 1}: Images saved to {batch_images_dir}")
    
    # 创建重叠批次（用于验证标注一致性）
    if batch_config.create_overlap:
        overlap_samples = create_overlap_batches(
            batches, 
            output_dir, 
            overlap_ratio=batch_config.overlap_ratio
        )
    
    # 保存批次总览
    batch_summary = {
        "total_batches": len(batches),
        "total_samples": len(failed_samples),
        "batch_config": batch_config.to_dict() if hasattr(batch_config, 'to_dict') else batch_config,
        "created_date": pd.Timestamp.now().isoformat(),
        "images_dir": images_dir,
        "batches": batch_metadata
    }
    
    summary_file = os.path.join(batches_dir, "batches_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2, default=str)
    
    # 生成批次分配建议
    generate_batch_assignment_guide(batches_dir, batch_metadata)
    
    print(f"✅ Created {len(batches)} experiment batches")
    print(f"📁 Batches saved to: {batches_dir}")
    print(f"📸 Images saved to: {images_dir}")
    
    return {
        "batches_dir": batches_dir,
        "images_dir": images_dir,
        "batch_files": batch_files,
        "summary_file": summary_file,
        "total_batches": len(batches),
        "total_samples": len(failed_samples)
    }

def create_overlap_batches(batches: List[List[Dict]], 
                          output_dir: str, 
                          overlap_ratio: float = 0.1) -> List[Dict]:
    """创建重叠批次用于验证标注一致性"""
    print(f"🔄 Creating overlap batches for consistency validation...")
    
    overlap_dir = os.path.join(output_dir, "overlap_validation")
    os.makedirs(overlap_dir, exist_ok=True)
    
    # 从每个批次中随机选择样本
    overlap_samples = []
    for batch_idx, batch in enumerate(batches):
        num_overlap = max(1, int(len(batch) * overlap_ratio))
        selected = random.sample(batch, min(num_overlap, len(batch)))
        
        for sample in selected:
            overlap_sample = sample.copy()
            overlap_sample["original_batch_id"] = batch_idx + 1
            overlap_sample["validation_type"] = "consistency_check"
            overlap_samples.append(overlap_sample)
    
    # 随机打乱重叠样本
    random.shuffle(overlap_samples)
    
    # 创建重叠验证批次
    overlap_batch_size = 50  # 重叠批次较小
    overlap_batches = []
    
    for i in range(0, len(overlap_samples), overlap_batch_size):
        batch_samples = overlap_samples[i:i + overlap_batch_size]
        
        overlap_filename = f"overlap_batch_{len(overlap_batches) + 1:02d}.jsonl"
        overlap_file_path = os.path.join(overlap_dir, overlap_filename)
        save_jsonl(overlap_file_path, batch_samples)
        
        overlap_batches.append({
            "filename": overlap_filename,
            "sample_count": len(batch_samples),
            "source_batches": list(set(s["original_batch_id"] for s in batch_samples))
        })
    
    # 保存重叠批次信息
    overlap_summary = {
        "total_overlap_samples": len(overlap_samples),
        "overlap_ratio": overlap_ratio,
        "overlap_batches": overlap_batches
    }
    
    with open(os.path.join(overlap_dir, "overlap_summary.json"), 'w') as f:
        json.dump(overlap_summary, f, indent=2)
    
    print(f"✅ Created {len(overlap_batches)} overlap validation batches")
    return overlap_samples

def generate_batch_assignment_guide(batches_dir: str, batch_metadata: List[Dict]):
    """生成批次分配指南"""
    print("📋 Generating batch assignment guide...")
    
    guide_content = f"""# Human Experiment Batch Assignment Guide

## Overview
Total batches: {len(batch_metadata)}
Total samples: {sum(b['sample_count'] for b in batch_metadata)}

## Batch Distribution Strategy

### Recommended Assignment Plan

#### For Individual Annotators (1 person per batch):
"""
    
    for i, batch in enumerate(batch_metadata):
        estimated_hours = batch['estimated_time_minutes'] / 60
        guide_content += f"""
**Batch {batch['batch_id']}** ({batch['filename']})
- Samples: {batch['sample_count']}
- Estimated time: {batch['estimated_time_minutes']:.0f} minutes ({estimated_hours:.1f} hours)
- Sequences: {len(batch['sequences'])} unique sequences
- Recommended for: {'Experienced annotator' if batch['sample_count'] > 150 else 'Any annotator'}
"""

    guide_content += f"""

#### For Team Annotation (Multiple annotators):
- **Phase 1**: Distribute batches 1-{len(batch_metadata)//2} to Team A
- **Phase 2**: Distribute batches {len(batch_metadata)//2 + 1}-{len(batch_metadata)} to Team B
- **Cross-validation**: Use overlap batches for quality control

### Quality Control Recommendations

1. **Training Phase**: Start with 10-20 samples from Batch 1
2. **Consistency Check**: Use overlap validation batches
3. **Progress Monitoring**: Review completed batches regularly
4. **Break Schedule**: Recommend 15-minute breaks every hour

### Batch Characteristics

| Batch ID | Samples | Est. Time (hrs) | Difficulty | Priority |
|----------|---------|-----------------|------------|----------|"""

    for batch in batch_metadata:
        hours = batch['estimated_time_minutes'] / 60
        difficulty = "High" if batch['sample_count'] > 200 else "Medium" if batch['sample_count'] > 100 else "Low"
        priority = "High" if batch['batch_id'] <= 3 else "Medium"
        guide_content += f"""
| {batch['batch_id']} | {batch['sample_count']} | {hours:.1f} | {difficulty} | {priority} |"""

    guide_content += """

### Instructions for Batch Processing

1. **Before Starting**:
   - Read the experiment guide thoroughly
   - Complete the training phase
   - Set up a comfortable working environment

2. **During Annotation**:
   - Take regular breaks (every 30-45 minutes)
   - Save progress frequently (auto-save enabled)
   - Note any issues or unclear cases

3. **After Completion**:
   - Review your annotations for obvious errors
   - Submit the completed batch
   - Provide feedback on batch difficulty

### Contact Information
For questions or technical issues, contact: [Your Contact Information]
"""

    guide_file = os.path.join(batches_dir, "batch_assignment_guide.md")
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"✅ Batch assignment guide saved to: {guide_file}")

def run_step6_preparation(config, step5_dir: str, dataset, 
                         experiment_config: Optional[HumanExperimentConfig] = None,
                         batch_config: Optional[Dict] = None) -> Dict[str, str]:
    """
    运行Step 6准备阶段：VLM失败样本筛选和分批实验数据准备
    """
    print("🔄 Starting Step 6: Human-Machine Cognitive Difference Exploration")
    
    if experiment_config is None:
        experiment_config = HumanExperimentConfig()
    
    # 创建输出目录
    output_dir = os.path.join(config.output_dir, "step6_human_experiment")
    os.makedirs(output_dir, exist_ok=True)
    
    results = {"output_dir": output_dir}
    
    # Step 6.1: 筛选VLM失败样本
    print("\n" + "-"*50)
    print("Phase 1: VLM Failed Sample Filtering")
    print("-"*50)
    
    failed_samples = filter_vlm_failed_samples(step5_dir, experiment_config)
    
    # 保存失败样本
    failed_samples_file = os.path.join(output_dir, "vlm_failed_samples.jsonl")
    save_jsonl(failed_samples_file, failed_samples)
    results["failed_samples_file"] = failed_samples_file
    
    # Step 6.2: 分析失败模式
    print("\n" + "-"*50)
    print("Phase 2: Failure Pattern Analysis")
    print("-"*50)
    
    failure_analysis = analyze_failure_patterns(failed_samples)
    
    # 保存分析结果
    analysis_file = os.path.join(output_dir, "vlm_failure_analysis.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(failure_analysis, f, ensure_ascii=False, indent=2)
    results["failure_analysis_file"] = analysis_file
    
    # 生成分析图表
    generate_failure_analysis_plots(failure_analysis, output_dir)
    
    # Step 6.3: 准备分批人类实验数据
    print("\n" + "-"*50)
    print("Phase 3: Batched Human Experiment Data Preparation")
    print("-"*50)
    
    batch_results = prepare_human_experiment_data_batched(
        failed_samples, dataset, output_dir, batch_config
    )
    results.update(batch_results)
    
    # 生成实验说明和IRB文档
    generate_experiment_documentation(output_dir, failure_analysis, len(failed_samples))
    
    print(f"\n✅ Step 6 preparation completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🧪 Ready for human experiment with {len(failed_samples)} samples in {batch_results['total_batches']} batches")
    print(f"📋 Next steps:")
    print(f"   1. Review batch assignment guide: {batch_results['batches_dir']}/batch_assignment_guide.md")
    print(f"   2. Distribute batches to annotators")
    print(f"   3. Run Gradio annotation tool for each batch")
    print(f"   4. Collect and merge results")
    
    return results

# 保持原有的其他函数不变...
def generate_failure_analysis_plots(analysis: Dict, output_dir: str, top_sequences: int = 20):
    """
    生成失败分析图表
    
    参数:
    - analysis: 分析结果字典
    - output_dir: 输出目录
    - top_sequences: 要显示的顶部序列数量，默认为20
    """
    print("📊 Generating failure analysis plots...")
    
    plots_dir = os.path.join(output_dir, "analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. 失败原因分布饼图 - 合并相似原因
    if analysis["failure_reasons"]:
        plt.figure(figsize=(10, 8))
        
        # 对失败原因进行分类汇总
        categorized_reasons = {
            "Low IoU (0.0-0.1)": 0,
            "Low IoU (0.1-0.2)": 0,
            "Low IoU (0.2-0.3)": 0,
            "Low IoU (0.3-0.4)": 0,
            "Low IoU (0.4-0.5)": 0,
            "Semantic Degradation": 0,
            "All Levels Failed": 0,
            "Other": 0
        }
        
        for reason, count in analysis["failure_reasons"].items():
            if reason == "semantic_degradation":
                categorized_reasons["Semantic Degradation"] += count
            elif reason == "all_levels_failed":
                categorized_reasons["All Levels Failed"] += count
            elif reason.startswith("low_iou_"):
                try:
                    iou_value = float(reason.split("_")[-1])
                    if 0.0 <= iou_value < 0.1:
                        categorized_reasons["Low IoU (0.0-0.1)"] += count
                    elif 0.1 <= iou_value < 0.2:
                        categorized_reasons["Low IoU (0.1-0.2)"] += count
                    elif 0.2 <= iou_value < 0.3:
                        categorized_reasons["Low IoU (0.2-0.3)"] += count
                    elif 0.3 <= iou_value < 0.4:
                        categorized_reasons["Low IoU (0.3-0.4)"] += count
                    elif 0.4 <= iou_value < 0.5:
                        categorized_reasons["Low IoU (0.4-0.5)"] += count
                    else:
                        categorized_reasons["Other"] += count
                except:
                    categorized_reasons["Other"] += count
            else:
                categorized_reasons["Other"] += count
        
        # 移除计数为0的类别
        categorized_reasons = {k: v for k, v in categorized_reasons.items() if v > 0}
        
        reasons = list(categorized_reasons.keys())
        counts = list(categorized_reasons.values())
        
        plt.pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90)
        plt.title('VLM Failure Reasons Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "failure_reasons_pie.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. IoU分布直方图 - 从failure_reasons中提取IoU值
    if analysis["failure_reasons"]:
        iou_values = []
        for reason, count in analysis["failure_reasons"].items():
            if reason.startswith("low_iou_"):
                try:
                    iou_value = float(reason.split("_")[-1])
                    # 为每个IoU值添加对应数量的样本
                    iou_values.extend([iou_value] * count)
                except:
                    pass
        
        if iou_values:
            plt.figure(figsize=(12, 6))
            plt.hist(iou_values, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('IoU Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('IoU Distribution of VLM Failed Samples', fontsize=16, fontweight='bold')
            plt.axvline(np.mean(iou_values), color='red', linestyle='--', 
                       label=f'Mean IoU: {np.mean(iou_values):.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "iou_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. 各层级性能对比
    if analysis["level_performance"]:
        levels = list(analysis["level_performance"].keys())
        success_rates = []
        avg_ious = []
        
        for level in levels:
            perf = analysis["level_performance"][level]
            success_rate = perf["success"] / perf["total"] if perf["total"] > 0 else 0
            success_rates.append(success_rate)
            avg_ious.append(perf["avg_iou"])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 成功率
        bars1 = ax1.bar(levels, success_rates, alpha=0.8)
        ax1.set_ylabel('Success Rate', fontsize=12)
        ax1.set_xlabel('Semantic Level', fontsize=12)
        ax1.set_title('VLM Success Rate by Semantic Level', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # 平均IoU
        bars2 = ax2.bar(levels, avg_ious, alpha=0.8, color='orange')
        ax2.set_ylabel('Average IoU', fontsize=12)
        ax2.set_xlabel('Semantic Level', fontsize=12)
        ax2.set_title('VLM Average IoU by Semantic Level', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for bar, iou in zip(bars2, avg_ious):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{iou:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "level_performance_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 序列分布条形图 - 显示指定数量的最常见序列
    if analysis["sequence_distribution"]:
        plt.figure(figsize=(15, 8))
        sequences = list(analysis["sequence_distribution"].keys())
        counts = list(analysis["sequence_distribution"].values())
        
        # 按数量排序
        sorted_data = sorted(zip(sequences, counts), key=lambda x: x[1], reverse=True)
        # 取指定数量的最常见序列
        top_n_sequences = sorted_data[:top_sequences]
        sequences, counts = zip(*top_n_sequences)
        
        plt.bar(range(len(sequences)), counts, alpha=0.8)
        plt.xlabel('Sequence', fontsize=12)
        plt.ylabel('Number of Failed Samples', fontsize=12)
        plt.title(f'Top {top_sequences} Sequences with Most VLM Failed Samples', fontsize=16, fontweight='bold')
        plt.xticks(range(len(sequences)), sequences, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "sequence_distribution.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Analysis plots saved to: {plots_dir}")

def generate_experiment_documentation(output_dir: str, analysis: Dict, num_samples: int):
    """生成实验文档和IRB申请材料"""
    print("📋 Generating experiment documentation...")
    
    docs_dir = os.path.join(output_dir, "experiment_docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # 1. 实验说明文档
    experiment_guide = f"""# Human-Machine Cognitive Difference Experiment Guide

## Experiment Overview
This experiment aims to explore the cognitive differences between humans and Vision-Language Models (VLMs) in visual object tracking tasks. We have identified {num_samples} cases where VLMs failed to correctly locate target objects despite having semantic descriptions.

## Experiment Setup
- **Total Samples**: {num_samples}
- **Task**: Locate target objects in images using provided semantic descriptions
- **Interface**: Web-based annotation tool (Gradio)
- **Time Estimate**: ~{num_samples * 0.5:.0f} minutes ({num_samples * 0.5 / 60:.1f} hours)
- **Data Organization**: Divided into manageable batches for better quality control

## Instructions for Participants

### Task Description
You will be shown pairs of images:
1. **Template Image**: Shows the target object in its initial appearance
2. **Current Image**: Contains the target object you need to locate

### Semantic Information Levels
You can choose to use different levels of semantic information:
1. **Visual Only**: Use only visual information from the template image
2. **+ Position**: Add position/location description
3. **+ Appearance**: Add appearance/visual features description  
4. **+ Dynamics**: Add motion/behavior description
5. **+ Context**: Add environmental context and distractor information
6. **Cannot Determine**: If the target cannot be reliably located

### How to Annotate
1. **Select Information Level**: Choose which level of information helps you most
2. **Draw Bounding Box**: Click and drag to draw a box around the target object
3. **Rate Confidence**: How confident are you in your annotation? (1-5 scale)
4. **Rate Difficulty**: How difficult was this case? (1-5 scale)
5. **Add Comments**: Optional feedback about the task

### Quality Guidelines
- Draw tight bounding boxes around the target object
- Be honest about your confidence and the information level used
- Add comments if you notice issues with the descriptions
- Take your time - accuracy is more important than speed

### Batch Processing Guidelines
- Work on one batch at a time
- Take breaks between batches (recommended: 15 minutes every hour)
- Save progress frequently (auto-save enabled)
- Complete batches in the suggested order when possible

## Data Privacy and Ethics
- All data is anonymized and used for research purposes only
- No personal information is collected
- Participation is voluntary and can be stopped at any time
- Results will be used to improve computer vision systems

## Contact Information
For questions about this experiment, please contact: [Your Contact Information]
"""

    with open(os.path.join(docs_dir, "experiment_guide.md"), 'w', encoding='utf-8') as f:
        f.write(experiment_guide)
    
    # 2. IRB申请摘要
    irb_summary = f"""# IRB Application Summary: Human-Machine Cognitive Difference Study

## Study Title
Exploring Semantic Cognition Differences Between Humans and Vision-Language Models in Visual Object Tracking

## Principal Investigator
[Your Name and Affiliation]

## Study Overview

### Background and Rationale
Vision-Language Models (VLMs) represent a significant advancement in AI, combining visual and textual understanding. However, their cognitive processes may differ significantly from human perception. This study investigates these differences in the context of visual object tracking, where both humans and VLMs must locate objects using semantic descriptions.

Our preliminary analysis of VLM performance identified {num_samples} challenging cases where state-of-the-art models failed to correctly locate target objects despite having detailed semantic descriptions. This presents an opportunity to understand:
1. How humans utilize different levels of semantic information
2. Where human cognition excels compared to current AI systems
3. How to improve AI systems based on human cognitive strategies

### Research Questions
1. **Primary**: Do humans perform better than VLMs when given the same semantic descriptions for visual object tracking?
2. **Secondary**: What levels of semantic information (position, appearance, dynamics, context) are most useful for humans vs. VLMs?
3. **Exploratory**: What characteristics make certain cases difficult for both humans and VLMs?

### Study Design

#### Participants
- **Target Sample Size**: 20-30 participants
- **Inclusion Criteria**: Adults (18+) with normal or corrected vision
- **Exclusion Criteria**: Vision impairments that affect object recognition
- **Recruitment**: Voluntary participation from [Your Institution/Platform]

#### Methodology
- **Design**: Comparative human-AI performance study
- **Task**: Web-based visual object localization using semantic descriptions
- **Data Collection**: Bounding box annotations, confidence ratings, information level preferences
- **Duration**: Approximately {num_samples * 0.5 / 60:.1f} hours per participant (divided into manageable batches)

#### Data Collection Procedure
1. **Consent**: Electronic informed consent before participation
2. **Training**: Brief tutorial on the annotation interface
3. **Main Task**: Annotate samples in batches using provided semantic descriptions
4. **Post-task**: Optional feedback survey

### Data and Privacy Protection

#### Data Collected
- **Visual Annotations**: Bounding box coordinates
- **Behavioral Data**: Information level preferences, confidence ratings, response times
- **Qualitative Data**: Optional comments and feedback
- **No Personal Data**: No names, emails, or identifying information collected

#### Data Security
- Data stored on secure servers with encryption
- Access limited to research team members
- Data retention: 5 years for research reproducibility
- Data sharing: Only aggregated, anonymous results will be published

#### Privacy Measures
- Participant assignment of anonymous IDs
- No linking between personal identity and responses
- Option to withdraw data at any time during participation

### Risk Assessment

#### Potential Risks
- **Minimal Risk Study**: No physical, psychological, or social risks beyond daily life
- **Eye Strain**: Possible minor eye fatigue from screen viewing
- **Time Commitment**: Approximately {num_samples * 0.5 / 60:.1f} hours of voluntary participation (divided into batches)

#### Risk Mitigation
- Clear instructions about taking breaks
- Voluntary participation with right to withdraw
- Task designed to minimize cognitive load
- Batch-based organization to prevent fatigue

### Benefits and Significance

#### Scientific Benefits
- Advance understanding of human vs. AI visual cognition
- Inform development of more human-like AI systems
- Contribute to computer vision and cognitive science literature

#### Societal Benefits
- Improve AI systems for applications like autonomous driving, medical imaging
- Better human-AI collaboration in visual tasks
- Enhanced AI safety through understanding failure modes

### Statistical Analysis Plan
- **Primary Analysis**: Compare human vs. VLM localization accuracy (IoU scores)
- **Secondary Analysis**: Analyze information level usage patterns
- **Exploratory Analysis**: Identify factors associated with task difficulty

### Ethical Considerations
- **Voluntary Participation**: Clear that participation is optional
- **Informed Consent**: Detailed explanation of study procedures
- **Data Minimization**: Collect only necessary data for research objectives
- **Transparency**: Results will be made publicly available

### Timeline
- **IRB Approval**: [Date]
- **Pilot Testing**: [Date range]
- **Data Collection**: [Date range] 
- **Analysis**: [Date range]
- **Publication**: [Date range]

## Statistical Power Analysis
Based on preliminary VLM results showing {analysis.get('level_performance', {}).get('1234', {}).get('avg_iou', 0):.3f} average IoU, we expect to detect meaningful differences in human performance with the proposed sample size.

## Data Management Plan
All data will be stored according to institutional data management policies, with regular backups and version control. Analysis code and aggregated results will be made available for reproducibility.

## Conclusion
This study addresses important questions about human-AI cognitive differences using a well-controlled experimental design with minimal risk to participants. The results will contribute to both scientific understanding and practical AI system development.
"""

    with open(os.path.join(docs_dir, "irb_application_summary.md"), 'w', encoding='utf-8') as f:
        f.write(irb_summary)
    
    # 3. 实验配置文件
    experiment_config = {
        "experiment_info": {
            "title": "Human-Machine Cognitive Difference Study",
            "version": "1.0",
            "created_date": pd.Timestamp.now().isoformat(),
            "total_samples": num_samples,
            "estimated_duration_minutes": num_samples * 0.5
        },
        "ui_config": {
            "show_confidence_rating": True,
            "show_difficulty_rating": True,
            "enable_comments": True,
            "auto_save_interval": 30,  # seconds
            "max_time_per_sample": 300  # seconds
        },
        "semantic_levels": {
            "1": "Visual Only (Template Image)",
            "2": "+ Position/Location Information", 
            "3": "+ Appearance/Visual Features",
            "4": "+ Motion/Behavior Dynamics",
            "5": "+ Environmental Context & Distractors",
            "6": "Cannot Determine Target"
        },
        "rating_scales": {
            "confidence": {
                "1": "Very Low Confidence",
                "2": "Low Confidence", 
                "3": "Medium Confidence",
                "4": "High Confidence",
                "5": "Very High Confidence"
            },
            "difficulty": {
                "1": "Very Easy",
                "2": "Easy",
                "3": "Medium",
                "4": "Difficult", 
                "5": "Very Difficult"
            }
        }
    }
    
    with open(os.path.join(docs_dir, "experiment_config.json"), 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Documentation saved to: {docs_dir}")
    print("📋 Generated files:")
    print("   - experiment_guide.md: Participant instructions")
    print("   - irb_application_summary.md: IRB application material")
    print("   - experiment_config.json: Technical configuration")