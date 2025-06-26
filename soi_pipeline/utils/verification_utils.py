# pytracking/soi_pipeline/utils/verification_utils.py
import os
import re
import json
import numpy as np
from typing import List, Optional, Tuple, Dict
import cv2

def load_jsonl(path: str) -> List[dict]:
    """加载JSONL文件"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def save_jsonl(path: str, records: List[dict]):
    """保存JSONL文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def xywh_to_xyxy(box: List[float]) -> List[float]:
    """将xywh格式转换为xyxy格式"""
    if not box or len(box) != 4:
        return None
    x, y, w, h = box
    return [x, y, x + w, y + h]

def xyxy_to_xywh(box: List[float]) -> List[float]:
    """将xyxy格式转换为xywh格式"""
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个边界框的IoU（xyxy格式）"""
    if not box1 or not box2:
        return 0.0
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def parse_vlm_output(text: str) -> Dict[str, str]:
    """解析VLM输出，提取不同级别的描述"""
    result = {
        "1": "",  # Position
        "2": "",  # Appearance
        "3": "",  # Dynamic state
        "4": ""   # Distractors
    }
    
    if not text:
        return result
    
    # 尝试解析结构化输出
    patterns = [
        (r"1\.\s*Position[^:]*:\s*([^\n]+)", "1"),
        (r"2\.\s*Appearance[^:]*:\s*([^\n]+)", "2"),
        (r"3\.\s*Dynamic[^:]*:\s*([^\n]+)", "3"),
        (r"4\.\s*Distractor[^:]*:\s*([^\n]+)", "4")
    ]
    
    for pattern, key in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            result[key] = match.group(1).strip()
    
    # 如果没有找到结构化格式，尝试其他格式
    if not any(result.values()):
        # 尝试寻找分段
        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('1.') or 'position' in line.lower():
                result["1"] = line
            elif line.startswith('2.') or 'appearance' in line.lower():
                result["2"] = line
            elif line.startswith('3.') or 'dynamic' in line.lower():
                result["3"] = line
            elif line.startswith('4.') or 'distractor' in line.lower():
                result["4"] = line
    
    return result

def parse_json_block(text: str) -> str:
    """
    提取```json代码块中的纯JSON内容
    如果没有markdown包裹，返回原始文本
    """
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def remap_bbox(bbox: List[float], input_w: int, input_h: int, raw_w: int, raw_h: int) -> List[int]:
    """将模型处理图上的bbox映射回原图坐标"""
    x1, y1, x2, y2 = bbox
    x1 = int(x1 / input_w * raw_w)
    x2 = int(x2 / input_w * raw_w)
    y1 = int(y1 / input_h * raw_h)
    y2 = int(y2 / input_h * raw_h)
    return [x1, y1, x2, y2]

def parse_detection_output(text: str) -> Optional[List[float]]:
    """
    从VLM输出中解析边界框（xyxy格式）

    支持格式：
    - [x1, y1, x2, y2]
    - x1, y1, x2, y2
    - <click>x1,y1</click>...
    - x1:..., y1:..., x2:..., y2:...
    - {"bbox": [x1, y1, x2, y2]}（嵌套JSON结构）
    """
    if not text:
        return None
    
    # Step 1: 提取 ```json ... ``` 块（去除markdown包裹）
    json_block_pattern = r"```json\s*(\[.*?\]|\{.*?\})\s*```"
    match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        json_text = match.group(1)
        try:
            parsed = json.loads(json_text)
            # 数组结构，取第一个 bbox/bbox_2d
            if isinstance(parsed, list) and len(parsed) > 0:
                first = parsed[0]
                box = first.get("bbox_2d") or first.get("bbox")
                if box and isinstance(box, list) and len(box) == 4:
                    return [float(x) for x in box]
            # 单个 dict 结构
            elif isinstance(parsed, dict):
                box = parsed.get("bbox_2d") or parsed.get("bbox")
                if box and isinstance(box, list) and len(box) == 4:
                    return [float(x) for x in box]
        except Exception as e:
            print(f"[⚠️] Failed to parse json block: {e}")

    # 正则提取（优先）
    patterns = [
        r"\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]",             # [x1, y1, x2, y2]
        r"<click>(\d+),\s*(\d+)</click>.*?<click>(\d+),\s*(\d+)</click>",           # <click>x1,y1</click>...
        r"x1:\s*(\d+).*?y1:\s*(\d+).*?x2:\s*(\d+).*?y2:\s*(\d+)",                   # x1: xx, y1: xx, ...
        r"(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)"                 # x1, y1, x2, y2
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                coords = [float(match.group(i)) for i in range(1, 5)]
                if coords[0] < coords[2] and coords[1] < coords[3]:
                    return coords
            except:
                continue

    # 如果正则没匹配，再尝试JSON结构
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "bbox" in data:
            box = data["bbox"]
            if isinstance(box, list) and len(box) == 4:
                coords = [float(x) for x in box]
                if coords[0] < coords[2] and coords[1] < coords[3]:
                    return coords
    except Exception:
        pass
    
    return None


def build_verification_prompt(vlm_desc: Dict[str, str], level: str = "1234", ref_mode: str = "none") -> str:
    """构建用于反向验证的提示词，拼接指定级别的描述为完整句子（保留原标点）"""
    
    # 拼接选定级别的描述内容
    description_parts = []
    if "1" in level and "level1" in vlm_desc:
        description_parts.append(vlm_desc["level1"].strip())
    if "2" in level and "level2" in vlm_desc:
        description_parts.append(vlm_desc["level2"].strip())
    if "3" in level and "level3" in vlm_desc:
        description_parts.append(vlm_desc["level3"].strip())
    if "4" in level and "level4" in vlm_desc:
        description_parts.append(vlm_desc["level4"].strip())

    # 拼接成一句完整的话
    full_description = " ".join(description_parts).strip()

    # 确保以句号结尾（如果没有）
    full_description = full_description.rstrip(".,!?。！？") + "."

    # 模板引导句
    if ref_mode == "first":
        prompt_intro = "You will be shown two images: the **template frame** (first image) and the **current frame** (second image). " \
                       "The green box in the template frame indicates the tracking target. Locate the target object in the current frame, based on the visual clue and the semantic description below.\n\n"
    elif ref_mode == "prev":
        prompt_intro = "You are given two images. The first image shows the target object at a previous time. " \
                       "Please locate the same object in the second image.\n\n"
    else:
        prompt_intro = "You are given one image containing the target and several distractors. " \
                       "Please identify and locate the target object based on the description.\n\n"

    # 完整提示词
    final_prompt = (
        prompt_intro +
        f"Description: {full_description}\n\n" +
        "Output its bbox coordinates using JSON format."
    )

    return final_prompt



def save_vis_image(image_path: str, gt_box: List[float], pred_box: Optional[List[float]], 
                  save_path: str, is_correct: bool, level: str):
    """保存可视化图像"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return
        
        # 绘制GT框（绿色）
        if gt_box:
            x1, y1, x2, y2 = [int(v) for v in gt_box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制预测框（正确为蓝色，错误为红色）
        if pred_box:
            x1, y1, x2, y2 = [int(v) for v in pred_box]
            color = (255, 0, 0) if is_correct else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Pred L{level}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 添加状态文本
        status_text = "CORRECT" if is_correct else "INCORRECT"
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 255, 0) if is_correct else (0, 0, 255), 2)
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        
    except Exception as e:
        print(f"Failed to save visualization: {e}")

def compute_verification_metrics(verified_records: List[dict], levels: List[str] = None) -> dict:
    """计算验证指标"""
    if levels is None:
        levels = ["12", "123", "1234"]
    
    metrics = {}
    for level in levels:
        correct = sum(1 for r in verified_records if r.get(f"verification_ok_{level}", False))
        total = len(verified_records)
        accuracy = correct / total if total > 0 else 0
        
        # 计算平均IoU
        ious = [r.get(f"verification_iou_{level}", 0) for r in verified_records]
        avg_iou = sum(ious) / len(ious) if ious else 0
        
        metrics[f"accuracy_{level}"] = accuracy
        metrics[f"avg_iou_{level}"] = avg_iou
        metrics[f"correct_{level}"] = correct
        metrics[f"total_{level}"] = total
    
    return metrics