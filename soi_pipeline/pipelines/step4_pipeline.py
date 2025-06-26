# pytracking/soi_pipeline/pipelines/step4_pipeline.py
import os
import json
import cv2
from tqdm import tqdm
from ..models.vlm_interface import VLMEngine, build_structured_prompt
from ..models.text_cleaner import TextCleaner
import numpy as np

def extract_category_from_path(image_path: str) -> str:
    """从图像路径提取类别名称"""
    path_parts = image_path.split(os.sep)
    for i in range(len(path_parts) - 1):
        current_part = path_parts[i]
        next_part = path_parts[i + 1]
        if next_part.startswith(current_part + "-"):
            return current_part
    return ""


def draw_gt_box_to_numpy(image_path: str, gt_box: list, box_color=(0, 255, 0), thickness=2) -> np.ndarray:
    """在图像上绘制绿色 GT 框并返回 numpy 格式图像"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    x1, y1, x2, y2 = map(int, gt_box)
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
    return image


def draw_gt_box(image_path: str, gt_box: list, box_color=(0, 255, 0), thickness=2) -> str:
    """在图像上绘制绿色 GT 框，并返回新图像的路径"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    x1, y1, x2, y2 = map(int, gt_box)
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
    
    # 构造保存路径（不保存到磁盘，只用于传入 VLM，可改为 numpy 直接传递）
    tmp_path = image_path.replace('.jpg', '_gt.jpg').replace('.png', '_gt.png')
    cv2.imwrite(tmp_path, image)
    return tmp_path


def run_step4_description_generation(config, dataset, step3_1_dir: str, step3_2_dir: str) -> str:
    """运行Step 4: VLM语义描述生成和文本清理"""
    print("🔄 Starting Step 4: VLM description generation and text cleaning")
    
    output_dir = os.path.join(config.output_dir, "step4_vlm_descriptions")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化VLM引擎
    try:
        vlm_engine = VLMEngine(config)
        print("✅ VLM engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize VLM engine: {e}")
        return output_dir
    
    # 初始化文本清理器（如果启用）
    text_cleaner = None
    if config.enable_text_cleaning:
        try:
            text_cleaner = TextCleaner(config.api_key)
            print("✅ Text cleaner initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize text cleaner: {e}")
            print("⚠️ Will proceed without text cleaning")
    
    # 创建数据集映射
    dataset_map = {seq.name: seq for seq in dataset}
    
    # 统计信息
    total_sequences = 0
    total_frames_processed = 0
    total_frames_cleaned = 0
    
    # 处理每个序列的SOI帧
    for soi_file in os.listdir(step3_2_dir):
        if not soi_file.endswith("_soi_frames.jsonl"):
            continue
        
        seq_name = soi_file.replace("_soi_frames.jsonl", "")
        if seq_name not in dataset_map:
            continue
        
        total_sequences += 1
        seq = dataset_map[seq_name]
        
        # 加载SOI帧索引
        try:
            with open(os.path.join(step3_2_dir, soi_file), 'r', encoding='utf-8') as f:
                soi_frame_indices = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load SOI frames for {seq_name}: {e}")
            continue
        
        if not soi_frame_indices:
            continue
        
        # 输出文件路径
        output_path = os.path.join(output_dir, f"{seq_name}_descriptions.jsonl")
        
        # 检查已处理的帧
        done_frames = set()
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        done_frames.add(data.get("frame_idx"))
                    except:
                        continue
        
        # 加载候选框文件
        boxes_file = os.path.join(step3_1_dir, f"{seq_name}.jsonl")
        if not os.path.exists(boxes_file):
            continue
        
        try:
            with open(boxes_file, 'r', encoding='utf-8') as f:
                all_frame_boxes = [json.loads(line.strip()) for line in f]
        except Exception as e:
            print(f"❌ Failed to load boxes for {seq_name}: {e}")
            continue
        
        # 处理每个SOI帧
        processed_count = 0
        cleaned_count = 0
        
        with open(output_path, 'a', encoding='utf-8') as f_out:
            for frame_idx in tqdm(soi_frame_indices, desc=f"Processing {seq_name}", leave=False):
                # 跳过已处理的帧
                if frame_idx in done_frames:
                    continue
                
                # 检查帧索引有效性
                if frame_idx >= len(seq.frames) or frame_idx >= len(all_frame_boxes):
                    continue
                
                current_frame_path = seq.frames[frame_idx]
                frame_boxes = all_frame_boxes[frame_idx]
                
                # 需要至少有GT + 1个干扰框
                if len(frame_boxes) < 2:
                    continue
                
                gt_box = frame_boxes[0]
                candidate_boxes = frame_boxes[1:]
                
                # 检查GT框是否过小
                try:
                    img = cv2.imread(current_frame_path)
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                        box_w = gt_box[2] - gt_box[0]
                        box_h = gt_box[3] - gt_box[1]
                        area_ratio = (box_w * box_h) / (img_w * img_h)
                        
                        # 使用配置中的阈值，如果没有则使用默认值
                        small_area_thresh = getattr(config, 'small_area_thresh', 0.001)
                        min_box_size = getattr(config, 'min_box_size', 10)
                        
                        if area_ratio < small_area_thresh or box_w < min_box_size or box_h < min_box_size:
                            continue
                except Exception:
                    continue
                
                # 提取类别信息
                category = extract_category_from_path(current_frame_path)
                
                # 构建提示词
                prompt = build_structured_prompt(
                    gt_box=gt_box,
                    candidate_boxes=candidate_boxes,
                    category=category,
                    template_mode=config.use_template
                )
                
                try:
                    # 目录结构：保存加框图像（例如 step4_vlm_descriptions/vis/sequence_name/frame_0010.jpg）
                    vis_dir = os.path.join(output_dir, "vis", seq_name)
                    os.makedirs(vis_dir, exist_ok=True)

                    # 当前帧加框并保存
                    current_frame_image = draw_gt_box_to_numpy(current_frame_path, gt_box)
                    current_frame_save_path = os.path.join(vis_dir, f"frame_{frame_idx:04d}.jpg")
                    cv2.imwrite(current_frame_save_path, current_frame_image)

                    # 默认只送当前帧
                    image_input = [current_frame_save_path]

                    # 模板帧加框并保存
                    if config.use_template:
                        template_image_path = seq.frames[0]
                        template_gt_box = seq.ground_truth_rect[0]
                        template_frame_image = draw_gt_box_to_numpy(template_image_path, template_gt_box)
                        template_frame_save_path = os.path.join(vis_dir, f"template.jpg")
                        cv2.imwrite(template_frame_save_path, template_frame_image)

                        image_input = [template_frame_save_path, current_frame_save_path]

                except Exception as e:
                    print(f"❌ Failed to prepare image input for {seq_name} frame {frame_idx}: {e}")
                    continue

                try:
                    # 生成描述
                    raw_description = vlm_engine.generate(image_input, prompt)
                    
                    if not raw_description.strip():
                        continue
                    
                    # 构建基础结果
                    result = {
                        "sequence_name": seq_name,
                        "frame_idx": frame_idx,
                        "image_path": current_frame_path,
                        "gt_box": gt_box,
                        "candidate_boxes": candidate_boxes,
                        "category": category,
                        "vlm_output_raw": raw_description,
                        "template_image": template_frame_save_path if config.use_template else None,
                        "prompt": prompt
                    }
                    
                    # 文本清理（如果启用）
                    cleaned_description = None
                    if text_cleaner and config.enable_text_cleaning:
                        try:
                            cleaned_description = text_cleaner.clean_text(raw_description)
                            if cleaned_description:
                                result["vlm_output_cleaned"] = cleaned_description
                                result["cleaning_status"] = "success"
                                cleaned_count += 1
                            else:
                                result["cleaning_status"] = "failed"
                                if config.verbose:
                                    print(f"⚠️ Text cleaning failed for {seq_name} frame {frame_idx}")
                        except Exception as e:
                            result["cleaning_status"] = "error"
                            result["cleaning_error"] = str(e)
                            if config.verbose:
                                print(f"❌ Text cleaning error for {seq_name} frame {frame_idx}: {e}")
                    else:
                        result["cleaning_status"] = "disabled"
                    
                    # 设置最终输出（优先使用清理后的文本）
                    if cleaned_description:
                        result["vlm_output"] = cleaned_description
                    else:
                        result["vlm_output"] = raw_description
                    
                    # 保存结果
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except Exception as e:
                    print(f"❌ Failed to process {seq_name} frame {frame_idx}: {e}")
                    continue
        
        total_frames_processed += processed_count
        total_frames_cleaned += cleaned_count
        
        if config.verbose and processed_count > 0:
            cleaning_info = f", cleaned {cleaned_count}" if config.enable_text_cleaning else ""
            print(f"✅ {seq_name}: processed {processed_count} frames{cleaning_info}")
    
    # 输出统计信息
    print(f"\n📊 Step 4 Statistics:")
    print(f"   Total sequences processed: {total_sequences}")
    print(f"   Total frames processed: {total_frames_processed}")
    if config.enable_text_cleaning:
        print(f"   Total frames cleaned: {total_frames_cleaned}")
        if total_frames_processed > 0:
            cleaning_rate = (total_frames_cleaned / total_frames_processed) * 100
            print(f"   Cleaning success rate: {cleaning_rate:.1f}%")
    
    print(f"✅ Step 4 completed. Results saved to {output_dir}")
    return output_dir


def validate_step4_output(output_dir: str, verbose: bool = False) -> dict:
    """验证Step 4输出的完整性"""
    print("🔍 Validating Step 4 output...")
    
    validation_results = {
        "total_files": 0,
        "total_records": 0,
        "records_with_cleaning": 0,
        "cleaning_success_rate": 0.0,
        "sequences": []
    }
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return validation_results
    
    # 检查所有描述文件
    for desc_file in os.listdir(output_dir):
        if not desc_file.endswith("_descriptions.jsonl"):
            continue
        
        validation_results["total_files"] += 1
        seq_name = desc_file.replace("_descriptions.jsonl", "")
        
        file_path = os.path.join(output_dir, desc_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                records = [json.loads(line.strip()) for line in f if line.strip()]
            
            seq_info = {
                "sequence_name": seq_name,
                "total_records": len(records),
                "records_with_raw": 0,
                "records_with_cleaned": 0,
                "cleaning_success": 0,
                "cleaning_failed": 0,
                "cleaning_error": 0
            }
            
            for record in records:
                validation_results["total_records"] += 1
                seq_info["total_records"] += 1
                
                if "vlm_output_raw" in record:
                    seq_info["records_with_raw"] += 1
                
                if "vlm_output_cleaned" in record:
                    seq_info["records_with_cleaned"] += 1
                    validation_results["records_with_cleaning"] += 1
                
                cleaning_status = record.get("cleaning_status", "unknown")
                if cleaning_status == "success":
                    seq_info["cleaning_success"] += 1
                elif cleaning_status == "failed":
                    seq_info["cleaning_failed"] += 1
                elif cleaning_status == "error":
                    seq_info["cleaning_error"] += 1
            
            validation_results["sequences"].append(seq_info)
            
            if verbose:
                print(f"   {seq_name}: {seq_info['total_records']} records, "
                      f"{seq_info['records_with_cleaned']} cleaned")
        
        except Exception as e:
            print(f"❌ Failed to validate {desc_file}: {e}")
    
    # 计算总体清理成功率
    if validation_results["total_records"] > 0:
        validation_results["cleaning_success_rate"] = (
            validation_results["records_with_cleaning"] / validation_results["total_records"]
        ) * 100
    
    print(f"✅ Validation completed:")
    print(f"   Total files: {validation_results['total_files']}")
    print(f"   Total records: {validation_results['total_records']}")
    print(f"   Records with cleaning: {validation_results['records_with_cleaning']}")
    print(f"   Overall cleaning rate: {validation_results['cleaning_success_rate']:.1f}%")
    
    return validation_results