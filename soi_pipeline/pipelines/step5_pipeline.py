# pytracking/soi_pipeline/pipelines/step5_pipeline.py
import os
import json
from tqdm import tqdm
from typing import Optional, List, Tuple
from dataclasses import dataclass

from ..models.vlm_interface import VLMEngine
from ..utils.verification_utils import (
    compute_iou, xywh_to_xyxy, parse_vlm_output, 
    parse_detection_output, build_verification_prompt,
    save_vis_image, parse_json_block, remap_bbox
)
from ..utils.visualization import draw_box_and_save
from PIL import Image, ImageDraw, ImageFont
import ast

@dataclass
class VerificationPipelineConfig:
    """反向验证管道配置"""
    vlm_config: dict
    vlm_model: str
    vlm_device: str
    vlm_batch_size: int
    vlm_max_length: int
    vlm_max_tokens: int
    vlm_temperature: float

@dataclass
class VerificationConfig:
    """反向验证配置"""
    enable_verification: bool = True
    ref_mode: str = "prev"  # none, first, prev
    levels: List[str] = None  # ["12", "123", "1234"]
    iou_threshold: float = 0.25
    save_visualizations: bool = False
    
    def __post_init__(self):
        if self.levels is None:
            self.levels = ["12", "123", "1234"]


def find_valid_template(sequence, current_idx: int) -> Tuple[Optional[str], Optional[list]]:
    """向前查找有效的模板帧"""
    for i in range(current_idx - 1, -1, -1):
        box = sequence.ground_truth_rect[i]
        if sum(box) > 0:
            return sequence.frames[i], box.tolist()
    return None, None


def run_step5_analysis(config, step5_dir: str) -> str:
    """
    运行Step 5的分析和可视化
    
    Args:
        config: 配置对象
        step5_dir: Step 5验证结果目录
    
    Returns:
        分析结果目录路径
    """
    from ..utils.step5_visualization import run_step5_visualization
    
    print("📊 Running Step 5 analysis and visualization...")
    
    # 创建分析输出目录
    analysis_dir = os.path.join(config.output_dir, "step5_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 运行可视化分析
    results = run_step5_visualization(
        input_dir=step5_dir,
        output_dir=analysis_dir,
        iou_threshold=config.verification_iou_threshold
    )
    
    print(f"✅ Step 5 analysis completed. Results saved to {analysis_dir}")
    return analysis_dir


def load_step4_descriptions(step4_dir: str, seq_name: str) -> List[dict]:
    """加载Step 4的描述文件"""
    desc_file = os.path.join(step4_dir, f"{seq_name}_descriptions.jsonl")
    
    if not os.path.exists(desc_file):
        return []
    
    try:
        with open(desc_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line.strip()) for line in f if line.strip()]
        return records
    except Exception as e:
        print(f"❌ Failed to load descriptions for {seq_name}: {e}")
        return []


def load_existing_verification_results(output_path: str, verification_levels: List[str]) -> Tuple[dict, set]:
    """
    加载已有的验证结果，找出需要重试的记录
    
    Returns:
        existing_records: 现有记录的字典 {frame_idx: record}
        frames_to_retry: 需要重试的帧索引集合
    """
    existing_records = {}
    frames_to_retry = set()
    
    if not os.path.exists(output_path):
        return existing_records, frames_to_retry
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line.strip())
                frame_idx = record.get("frame_idx")
                if frame_idx is None:
                    continue
                
                existing_records[frame_idx] = record
                
                # 检查是否需要重试
                needs_retry = False
                for level in verification_levels:
                    # 如果有错误，需要重试
                    if f"verification_error_{level}" in record:
                        needs_retry = True
                        break
                    # 如果缺少这个level的验证结果，需要重试
                    if f"verification_ok_{level}" not in record:
                        needs_retry = True
                        break
                    # 如果缺少这个level的预测结果，需要重试
                    if f"verification_pred_{level}" not in record:
                        needs_retry = True
                        break

                if needs_retry:
                    frames_to_retry.add(frame_idx)
                    
    except Exception as e:
        print(f"⚠️ Error loading existing results: {e}")
        # 如果读取失败，返回空结果，重新处理所有帧
        return {}, set()
    
    return existing_records, frames_to_retry


def merge_verification_results(existing_record: dict, new_verification: dict, level: str) -> dict:
    """合并验证结果，保留原有成功的结果，更新失败或新的结果"""
    # 如果原记录中这个level已经成功，保留原结果
    if existing_record.get(f"verification_ok_{level}", False):
        return existing_record
    
    # 否则更新为新结果
    verification_keys = [
        f"verification_ok_{level}",
        f"verification_pred_{level}", 
        f"verification_output_{level}",
        f"verification_iou_{level}",
        f"verification_prompt_{level}",
        f"verification_error_{level}"
    ]
    
    for key in verification_keys:
        if key in new_verification:
            existing_record[key] = new_verification[key]
        elif key.endswith("_error") and key in existing_record:
            # 如果新验证没有错误，删除旧的错误信息
            del existing_record[key]
    
    return existing_record


def run_reverse_verification(config, step4_dir: str, dataset, verification_config: VerificationConfig) -> str:
    """运行反向验证功能"""
    print("🔄 Starting reverse verification...")
    
    output_dir = os.path.join(config.output_dir, "step5_verification_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化VLM引擎
    try:
        vlm_engine = VLMEngine(config)
        print("✅ VLM engine initialized for verification")
    except Exception as e:
        print(f"❌ Failed to initialize VLM engine: {e}")
        return output_dir
    
    # 创建序列名到序列对象的映射
    seq_map = {seq.name: seq for seq in dataset}
    
    # 统计信息
    total_sequences = 0
    total_records = 0
    total_verified = 0
    total_retried = 0
    verification_stats = {level: {"total": 0, "passed": 0} for level in verification_config.levels}
    
    # 获取所有需要验证的序列
    description_files = [f for f in os.listdir(step4_dir) if f.endswith("_descriptions.jsonl")]
    
    # 处理每个序列
    for desc_file in description_files:
        seq_name = desc_file.replace("_descriptions.jsonl", "")
        if seq_name == 'helmet-11':
            print(f"⚠️ Sequence {seq_name} debug")
        if seq_name not in seq_map:
            print(f"⚠️ Sequence {seq_name} not found in dataset, skip")
            continue
        
        total_sequences += 1
        seq = seq_map[seq_name]
        
        # 输出文件路径
        output_file = f"{seq_name}_verified.jsonl"
        output_path = os.path.join(output_dir, output_file)

        # 加载已有的验证结果
        existing_records, frames_to_retry = load_existing_verification_results(
            output_path, verification_config.levels
        )

        # # 检查是否已处理
        # if os.path.exists(output_path):
        #     print(f"⚡️ Verification already exists for {seq_name}, skip")
        #     continue
        
        # 加载Step 4的描述
        records = load_step4_descriptions(step4_dir, seq_name)
        if not records:
            print(f"⚠️ No descriptions found for {seq_name}")
            continue
        
        total_records += len(records)
        
        # 验证每条记录
        # verified_records = []

        # 确定需要处理的记录
        records_to_process = []
        for record in records:
            frame_idx = record["frame_idx"]
            if frame_idx not in existing_records or frame_idx in frames_to_retry:
                records_to_process.append(record)

        if not records_to_process:
            print(f"⚡️ All verifications completed for {seq_name}, skip")
            continue

        print(f"📝 Processing {seq_name}: {len(records_to_process)} frames to verify "
              f"({len(frames_to_retry)} retries, {len(records_to_process) - len(frames_to_retry)} new)")
        
        seq_verified_count = 0
        seq_retried_count = 0

        for record in tqdm(records_to_process, desc=f"Verifying {seq_name}", leave=False):
            frame_idx = record["frame_idx"]
            img_path = seq.frames[frame_idx]
            gt_box = record["gt_box"]
            
            # 如果是重试，增加计数
            if frame_idx in frames_to_retry:
                seq_retried_count += 1
                total_retried += 1

        # for record in tqdm(records, desc=f"Verifying {seq_name}", leave=False):
        #     frame_idx = record["frame_idx"]
        #     img_path = seq.frames[frame_idx]
        #     gt_box = record["gt_box"]
            
            # 获取VLM输出
            vlm_output = record.get("vlm_output")
            vlm_output_raw = record.get("vlm_output_raw", "")
            # 跳过空结果
            if not vlm_output and not vlm_output_raw:
                print(f"⚠️ No VLM output found for {seq_name} frame {frame_idx}")
                continue

            # 解析结构化描述
            if isinstance(vlm_output, dict):
                vlm_parsed = vlm_output  # 已清洗
            elif isinstance(vlm_output, str) and vlm_output.strip().startswith("{"):
                try:
                    vlm_parsed = json.loads(vlm_output)
                except Exception as e:
                    print(f"⚠️ Failed to parse vlm_output as JSON for {seq_name} frame {frame_idx}: {e}")
                    vlm_parsed = parse_vlm_output(vlm_output_raw)  # fallback
            else:
                vlm_parsed = parse_vlm_output(vlm_output or vlm_output_raw)  # fallback 原始格式
            
            # 选择参考帧
            template_path = None
            template_box = None
            
            if verification_config.ref_mode == "first":
                if len(seq.frames) > 0 and sum(seq.ground_truth_rect[0]) > 0:
                    template_path = seq.frames[0]
                    template_box = xywh_to_xyxy(seq.ground_truth_rect[0].tolist())
            elif verification_config.ref_mode == "prev":
                template_path, template_box_xywh = find_valid_template(seq, frame_idx)
                if template_box_xywh:
                    template_box = xywh_to_xyxy(template_box_xywh)

            # 如果这是现有记录，使用它作为基础
            if frame_idx in existing_records:
                current_record = existing_records[frame_idx].copy()
            else:
                current_record = record.copy()

            # # 添加验证相关的元数据
            # record["verification_config"] = {
            #     "ref_mode": verification_config.ref_mode,
            #     "levels": verification_config.levels,
            #     "iou_threshold": verification_config.iou_threshold
            # }

            # 添加验证相关的元数据
            current_record["verification_config"] = {
                "ref_mode": verification_config.ref_mode,
                "levels": verification_config.levels,
                "iou_threshold": verification_config.iou_threshold
            }

            if template_path:
                record["template_path"] = template_path
                record["template_box"] = template_box
            
            # 对每个级别进行验证
            verification_success = False
            for level in verification_config.levels:
                # 如果这个level已经成功，跳过
                # 如果既没有 verification_ok 也没有 verification_error，说明没跑过，需要处理
                if f"verification_pred_{level}" in current_record or f"verification_error_{level}" in current_record:
                    continue  # 跳过已经处理过的，无论成功或失败
                # if not current_record.get(f"verification_error_{level}", None):
                #     continue

                verification_stats[level]["total"] += 1
                
                prompt = build_verification_prompt(vlm_parsed, level, ref_mode=verification_config.ref_mode)
                
                # 准备图像输入
                if template_path and template_box and verification_config.ref_mode != "none":
                    # 临时绘制GT框后的模板图像
                    vis_dir = os.path.join(output_dir, "temp_templates", seq_name)
                    os.makedirs(vis_dir, exist_ok=True)
                    if verification_config.ref_mode == "first":
                        template_vis_path = os.path.join(vis_dir, f"template_frame.jpg")
                    else:
                        template_vis_path = os.path.join(vis_dir, f"template_frame_{frame_idx}.jpg")
                    
                    try:
                        draw_box_and_save(template_path, template_box, template_vis_path, color=(0, 255, 0), label="GT")  # 自定义工具
                        image_inputs = [template_vis_path, img_path]
                    except Exception as e:
                        print(f"⚠️ Failed to draw GT on template for {seq_name} frame {frame_idx}: {e}")
                        image_inputs = [template_path, img_path]
                else:
                    image_inputs = [img_path]
                
                # 运行推理
                try:
                    pred_text, input_w, input_h = vlm_engine.generate(
                        image_paths=image_inputs,
                        prompt=prompt,
                    )
                    
                    # 解析预测框
                    pred_box = parse_json_block(pred_text)
                    if 'none' in pred_box:
                        pred_box = [0, 0, 0, 0]
                        is_correct = False
                        iou_val = 0.0
                    else:
                        json_output = ast.literal_eval(pred_box)
                        for i, bounding_box in enumerate(json_output):
                            if bounding_box:
                                raw_image = Image.open(img_path)
                                raw_w, raw_h = raw_image.size
                                # Convert normalized coordinates to absolute coordinates
                                abs_y1 = int(bounding_box["bbox_2d"][1]/input_h * raw_h)
                                abs_x1 = int(bounding_box["bbox_2d"][0]/input_w * raw_w)
                                abs_y2 = int(bounding_box["bbox_2d"][3]/input_h * raw_h)
                                abs_x2 = int(bounding_box["bbox_2d"][2]/input_w * raw_w)
                                if abs_x1 > abs_x2:
                                    abs_x1, abs_x2 = abs_x2, abs_x1
                                if abs_y1 > abs_y2:
                                    abs_y1, abs_y2 = abs_y2, abs_y1

                                pred_box = [abs_x1, abs_y1, abs_x2, abs_y2]
                        # 计算IoU
                        iou_val = compute_iou(gt_box, pred_box) if pred_box else 0.0
                        is_correct = iou_val >= verification_config.iou_threshold
                    
                    if is_correct:
                        verification_stats[level]["passed"] += 1
                        verification_success = True
                    
                    # 保存结果
                    current_record[f"verification_ok_{level}"] = is_correct
                    current_record[f"verification_pred_{level}"] = pred_box
                    current_record[f"verification_output_{level}"] = pred_text
                    current_record[f"verification_iou_{level}"] = iou_val
                    current_record[f"verification_prompt_{level}"] = prompt

                    # 删除可能存在的错误信息
                    if f"verification_error_{level}" in current_record:
                        del current_record[f"verification_error_{level}"]

                    # 保存可视化（如果启用）
                    if verification_config.save_visualizations:
                        vis_dir = os.path.join(output_dir, "visualizations", seq_name)
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_path = os.path.join(vis_dir, f"frame_{frame_idx}_level_{level}.jpg")
                        save_vis_image(img_path, gt_box, pred_box, vis_path, is_correct, level)
                    
                except Exception as e:
                    print(f"❌ Verification failed for {seq_name} frame {frame_idx} level {level}: {e}")
                    # record[f"verification_ok_{level}"] = False
                    record[f"verification_error_{level}"] = str(e)
            
            if verification_success:
                seq_verified_count += 1
                total_verified += 1

            # 更新existing_records
            existing_records[frame_idx] = current_record

            # verified_records.append(record)
        
        # # 保存验证结果
        # try:
        #     with open(output_path, 'w', encoding='utf-8') as f:
        #         for record in verified_records:
        #             f.write(json.dumps(record, ensure_ascii=False) + '\n')
        # except Exception as e:
        #     print(f"❌ Failed to save verification results for {seq_name}: {e}")
        #     continue
        
        # 保存所有验证结果（包括之前成功的和新处理的）
        try:
            # 创建临时文件
            temp_path = output_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                # 按frame_idx排序后写入
                for frame_idx in sorted(existing_records.keys()):
                    f.write(json.dumps(existing_records[frame_idx], ensure_ascii=False) + '\n')
            
            # 原子性替换文件
            os.replace(temp_path, output_path)
            
        except Exception as e:
            print(f"❌ Failed to save verification results for {seq_name}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue

        if config.verbose:
            total_frames = len(existing_records)
            success_frames = sum(1 for r in existing_records.values() 
                               if any(r.get(f"verification_ok_{level}", False) 
                                     for level in verification_config.levels))
            success_rate = (success_frames / total_frames) * 100 if total_frames > 0 else 0
            print(f"✅ {seq_name}: processed {len(records_to_process)} frames "
                  f"({seq_retried_count} retried), "
                  f"total {success_frames}/{total_frames} successful ({success_rate:.1f}%)")
    
    # 输出统计信息
    print(f"\n📊 Step 5 Verification Statistics:")
    print(f"   Total sequences: {total_sequences}")
    print(f"   Total records: {total_records}")
    print(f"   Records retried: {total_retried}")
    print(f"   Records with at least one successful verification: {total_verified}")
    
    if total_records > 0:
        overall_success_rate = (total_verified / total_records) * 100
        print(f"   Overall success rate: {overall_success_rate:.1f}%")
    
    print(f"\n📋 Verification by Level:")
    for level in verification_config.levels:
        level_total = verification_stats[level]["total"]
        level_passed = verification_stats[level]["passed"]
        if level_total > 0:
            level_rate = (level_passed / level_total) * 100
            print(f"   Level {level}: {level_passed}/{level_total} ({level_rate:.1f}%)")
    
    print(f"\n✅ Reverse verification completed. Results saved to {output_dir}")
    return output_dir


def run_step5_pipeline(config, step4_dir: str, dataset, 
                      verification_config: Optional[VerificationConfig] = None,
                      run_analysis: bool = True) -> dict:
    """
    运行Step 5: 反向验证
    
    Args:
        config: 主配置对象
        step4_dir: Step 4输出目录（包含描述和清理结果）
        dataset: 数据集对象
        verification_config: 验证配置
        run_analysis: 是否运行分析
    
    Returns:
        包含输出目录的字典
    """
    print("🔄 Starting Step 5: Reverse Verification")
    
    if dataset is None:
        print("❌ Dataset is required for verification")
        return {}
    
    if verification_config is None:
        verification_config = VerificationConfig()
    
    # 验证Step 4输出目录
    if not os.path.exists(step4_dir):
        print(f"❌ Step 4 output directory not found: {step4_dir}")
        return {}
    
    # 检查Step 4输出文件
    description_files = [f for f in os.listdir(step4_dir) if f.endswith("_descriptions.jsonl")]
    if not description_files:
        print(f"❌ No description files found in {step4_dir}")
        return {}
    
    print(f"📂 Found {len(description_files)} description files to verify")
    print(f"🔧 Verification configuration:")
    print(f"   Reference mode: {verification_config.ref_mode}")
    print(f"   Verification levels: {verification_config.levels}")
    print(f"   IoU threshold: {verification_config.iou_threshold}")
    print(f"   Save visualizations: {verification_config.save_visualizations}")
    
    results = {}
    
    # 运行反向验证
    verification_dir = run_reverse_verification(
        config, step4_dir, dataset, verification_config
    )
    results["verification_dir"] = verification_dir
    
    # 运行分析和可视化
    if run_analysis and verification_dir:
        print("\n" + "-"*50)
        print("Phase 2: Analysis and Visualization")
        print("-"*50)
        
        analysis_dir = run_step5_analysis(config, verification_dir)
        results["analysis_dir"] = analysis_dir
    
    print(f"\n✅ Step 5 completed successfully!")
    return results


def validate_step5_output(output_dir: str, verbose: bool = False) -> dict:
    """验证Step 5输出的完整性"""
    print("🔍 Validating Step 5 output...")
    
    validation_results = {
        "total_files": 0,
        "total_records": 0,
        "verification_levels": [],
        "level_statistics": {},
        "sequences": []
    }
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return validation_results
    
    # 检查所有验证文件
    for verified_file in os.listdir(output_dir):
        if not verified_file.endswith("_verified.jsonl"):
            continue
        
        validation_results["total_files"] += 1
        seq_name = verified_file.replace("_verified.jsonl", "")
        
        file_path = os.path.join(output_dir, verified_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                records = [json.loads(line.strip()) for line in f if line.strip()]
            
            seq_info = {
                "sequence_name": seq_name,
                "total_records": len(records),
                "level_results": {}
            }
            
            # 分析每条记录
            for record in records:
                validation_results["total_records"] += 1
                
                # 检查验证级别
                for key in record.keys():
                    if key.startswith("verification_ok_"):
                        level = key.replace("verification_ok_", "")
                        if level not in validation_results["verification_levels"]:
                            validation_results["verification_levels"].append(level)
                        
                        if level not in seq_info["level_results"]:
                            seq_info["level_results"][level] = {"total": 0, "passed": 0}
                        
                        seq_info["level_results"][level]["total"] += 1
                        if record[key]:
                            seq_info["level_results"][level]["passed"] += 1
            
            validation_results["sequences"].append(seq_info)
            
            if verbose:
                print(f"   {seq_name}: {seq_info['total_records']} records verified")
        
        except Exception as e:
            print(f"❌ Failed to validate {verified_file}: {e}")
    
    # 计算总体统计
    for level in validation_results["verification_levels"]:
        validation_results["level_statistics"][level] = {"total": 0, "passed": 0}
        
        for seq_info in validation_results["sequences"]:
            if level in seq_info["level_results"]:
                validation_results["level_statistics"][level]["total"] += seq_info["level_results"][level]["total"]
                validation_results["level_statistics"][level]["passed"] += seq_info["level_results"][level]["passed"]
    
    print(f"✅ Validation completed:")
    print(f"   Total files: {validation_results['total_files']}")
    print(f"   Total records: {validation_results['total_records']}")
    print(f"   Verification levels: {validation_results['verification_levels']}")
    
    for level, stats in validation_results["level_statistics"].items():
        if stats["total"] > 0:
            success_rate = (stats["passed"] / stats["total"]) * 100
            print(f"   Level {level}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
    
    return validation_results


# 保持向后兼容的包装函数
def run_step5_verification_only(config, step4_dir: str, dataset, verification_config: VerificationConfig = None) -> str:
    """向后兼容的函数，只运行验证"""
    results = run_step5_pipeline(
        config, step4_dir, dataset,
        verification_config=verification_config,
        run_analysis=False
    )
    return results.get("verification_dir", "")