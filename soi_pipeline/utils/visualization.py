# pytracking/soi_pipeline/utils/visualization.py
import os
import cv2
import numpy as np
from typing import List, Optional


def draw_bounding_boxes(img: np.ndarray, gt_box: List[float], 
                       candidate_boxes: List[List[float]], 
                       labels: Optional[List[str]] = None) -> np.ndarray:
    """在图像上绘制边界框"""
    img_copy = img.copy()
    
    # 绘制GT框（绿色）
    x1, y1, x2, y2 = [int(coord) for coord in gt_box]
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_copy, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 绘制候选框（红色）
    for i, box in enumerate(candidate_boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        label = labels[i] if labels and i < len(labels) else f"C{i+1}"
        cv2.putText(img_copy, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return img_copy


def draw_box_and_save(image_path: str, box_xyxy: List[float], save_path: str, 
                      color: tuple = (0, 255, 0), label: Optional[str] = "GT") -> None:
    """
    在图像上绘制一个边界框并保存到文件。

    Args:
        image_path: 输入图像路径。
        box_xyxy: 边界框 [x1, y1, x2, y2]。
        save_path: 保存路径。
        color: 框的颜色 (B, G, R)，默认绿色。
        label: 可选的文本标签。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    x1, y1, x2, y2 = [int(c) for c in box_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    if label:
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite(save_path, img)
    

def create_description_visualization(image_path: str, gt_box: List[float], 
                                   candidate_boxes: List[List[float]], 
                                   description: str, output_path: str,
                                   show_description: bool = True) -> bool:
    """创建带描述的可视化图像"""
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to load image {image_path}")
            return False
        
        # 绘制边界框
        img_with_boxes = draw_bounding_boxes(img, gt_box, candidate_boxes)
        
        # 添加描述文本
        if show_description and description:
            # 尝试解析JSON描述
            try:
                import json
                desc_dict = json.loads(description)
                desc_lines = [f"{k}: {v}" for k, v in desc_dict.items()]
            except:
                # 如果不是JSON，则按句号分割
                desc_lines = description.split('. ')
                desc_lines = [line[:80] + '...' if len(line) > 80 else line for line in desc_lines]
            
            # 绘制文本背景
            text_height = 25
            total_height = len(desc_lines) * text_height + 20
            cv2.rectangle(img_with_boxes, (10, 10), 
                         (min(img.shape[1] - 10, 800), total_height), 
                         (0, 0, 0), -1)
            cv2.rectangle(img_with_boxes, (10, 10), 
                         (min(img.shape[1] - 10, 800), total_height), 
                         (255, 255, 255), 2)
            
            # 绘制文本
            for i, line in enumerate(desc_lines):
                if i * text_height + 35 > img.shape[0] - 10:  # 避免超出图像边界
                    break
                y_pos = 35 + i * text_height
                cv2.putText(img_with_boxes, line[:70], (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存图像
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, img_with_boxes)
        
        if not success:
            print(f"Warning: Failed to save visualization to {output_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Warning: Failed to create visualization: {e}")
        return False