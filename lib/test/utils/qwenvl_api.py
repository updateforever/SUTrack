from typing import Dict, Optional, List, Tuple
import requests
import base64
from PIL import Image
from io import BytesIO
import json
import os
import ast
import xml.etree.ElementTree as ET
from openai import OpenAI
import re
import numpy as np
import cv2


class GroundingIntegration:
    """
    完整的VLM Grounding集成类 - 基于SOI文本标签的帧检测和VLM API调用
    """
    
    def __init__(self,
                 api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 api_type: str = "qwenvl",
                 api_key: Optional[str] = None,
                 confidence_threshold: float = 0.4,
                 fusion_weight: float = 0.7,
                 min_iou_threshold: float = 0.2,
                 text_effective_frames: int = 30,
                 text_base_path: str = "/home/wyp/project/SUTrack/soi_outputs/lasot_old/step4_vlm_descriptions",
                 qwen_model_id: str = "qwen2.5-vl-72b-instruct",
                 qwen_min_pixels: int = 512*28*28,
                 qwen_max_pixels: int = 2048*28*28):
        
        self.api_url = api_url
        self.api_type = api_type.lower()
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self.fusion_weight = fusion_weight
        self.min_iou_threshold = min_iou_threshold
        self.text_effective_frames = text_effective_frames
        self.text_base_path = text_base_path
        self.qwen_model_id = qwen_model_id
        self.qwen_min_pixels = qwen_min_pixels
        self.qwen_max_pixels = qwen_max_pixels
        
        # SOI文本标签相关状态
        self.text_descriptions = {}
        self.current_text_frame = None
        self.current_text = None
        self.text_loaded = False
        
        # Grounding状态跟踪
        self.last_grounding_frame = -1
        self.grounding_count = 0
        self.current_grounding_text = None
        self.grounding_text_frame = None
        self.grounding_history = []
        
        # 初始化API客户端
        self._init_api_client()
        
        print(f"Grounding initialized - API: {api_type}, threshold: {confidence_threshold}")
        print(f"Text path: {text_base_path}, effective frames: {text_effective_frames}")

    def load_text_data(self, sequence_name: str) -> bool:
        """加载序列文本数据 - 仿照TextLabelIntegration"""
        if self.text_loaded:
            return True
            
        jsonl_file = os.path.join(self.text_base_path, f"{sequence_name}_descriptions.jsonl")
        
        if not os.path.exists(jsonl_file):
            print(f"SOI text file not found: {jsonl_file}")
            self.text_loaded = True
            return False
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    frame_idx = data.get('frame_idx')
                    vlm_output = data.get('vlm_output_cleaned') or data.get('vlm_output')
                    
                    if frame_idx is not None and vlm_output is not None:
                        self.text_descriptions[frame_idx] = vlm_output
            
            self.text_loaded = True
            print(f"Loaded {len(self.text_descriptions)} SOI text descriptions for {sequence_name}")
            return True
            
        except Exception as e:
            print(f"Error loading SOI text: {e}")
            self.text_loaded = True
            return False

    def get_text_for_frame(self, frame_id: int) -> Optional[str]:
        """获取当前帧的文本描述 - 仿照TextLabelIntegration"""
        # 检查当前帧是否有新文本
        if frame_id in self.text_descriptions:
            self.current_text_frame = frame_id
            vlm_output = self.text_descriptions[frame_id]
            self.current_text = self._extract_text_from_vlm_output(vlm_output)
            return self.current_text
        
        # 检查保质期
        if (self.current_text_frame is not None and 
            self.current_text is not None and
            frame_id <= self.current_text_frame + self.text_effective_frames):
            return self.current_text
        
        # 超过保质期，清除
        self.current_text_frame = None
        self.current_text = None
        return None

    def _extract_text_from_vlm_output(self, vlm_output: Dict) -> str:
        """从VLM输出中提取文本描述"""
        if not vlm_output:
            return ""
        
        text_parts = []
        
        # 提取所有级别的文本
        if 'level1' in vlm_output:
            text_parts.append(vlm_output['level1'].strip())
        
        if 'level2' in vlm_output:
            text_parts.append(vlm_output['level2'].strip())
        
        if 'level3' in vlm_output:
            text_parts.append(vlm_output['level3'].strip())
        
        if 'level4' in vlm_output:
            level4 = vlm_output['level4']
            if isinstance(level4, list):
                text_parts.extend([t.strip() for t in level4 if t.strip()])
            elif isinstance(level4, str):
                text_parts.append(level4.strip())
        
        return ' '.join(text_parts)

    def is_soi_frame(self, frame_id: int, confidence: float) -> bool:
        """
        判断是否为SOI帧 - 基于文本标签的新逻辑
        只有当 有文本标签 且 跟踪置信度低于阈值 时，才调用VLM修正
        """
        # 获取当前帧的文本标签
        current_text = self.get_text_for_frame(frame_id)
        
        # 没有文本标签，不是SOI帧
        if not current_text:
            return False
        
        # 有文本标签且置信度低于阈值，才是SOI帧
        low_confidence = confidence < self.confidence_threshold
        not_first_frame = frame_id > 1
        
        if not_first_frame and low_confidence:
            print(f"Frame {frame_id}: SOI detected - text available, low confidence ({confidence:.3f})")
            return True
        
        return False

    def get_current_text(self, frame_id: int, default_text: str) -> str:
        """获取当前文本描述用于grounding"""
        # 优先使用SOI文本标签
        soi_text = self.get_text_for_frame(frame_id)
        if soi_text:
            return soi_text
        
        # 否则使用默认文本
        return default_text

    def _init_api_client(self):
        """初始化API客户端"""
        if self.api_type == "qwenvl":
            if not self.api_key:
                print("Warning: No API key found for Qwen-VL")
                return
            
            try:
                self.qwen_client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                print("Qwen-VL client initialized")
            except Exception as e:
                print(f"Failed to initialize Qwen client: {e}")

    def _encode_image_base64(self, image: np.ndarray) -> str:
        """图像编码为base64"""
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image: {e}")
            return ""

    def call_grounding_api(self, image: np.ndarray, text_description: str) -> Optional[List[float]]:
        """调用grounding API"""
        try:
            if self.api_type == "qwenvl":
                return self._call_qwenvl_api(image, text_description)
            else:
                print(f"Unsupported API type: {self.api_type}")
                return None
        except Exception as e:
            print(f"Grounding API call failed: {e}")
            return None

    def _call_qwenvl_api(self, image: np.ndarray, text_description: str) -> Optional[List[float]]:
        """Qwen-VL API调用"""
        try:
            if not hasattr(self, 'qwen_client'):
                return None
            
            # 编码图像
            image_base64 = self._encode_image_base64(image)
            if not image_base64:
                return None
            
            height, width = image.shape[:2]
            
            # 构建prompt
            grounding_prompt = f"""Please locate the '{text_description}' in the image and return the bounding box coordinates. Return the result in JSON format."""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                        {"type": "text", "text": grounding_prompt},
                    ],
                }
            ]
            
            # 调用API
            completion = self.qwen_client.chat.completions.create(
                model=self.qwen_model_id,
                messages=messages,
                temperature=0.1,
            )
            
            response_content = completion.choices[0].message.content
            return self._parse_qwenvl_response(response_content, width, height)
            
        except Exception as e:
            print(f"Qwen-VL API error: {e}")
            return None

    def _parse_qwenvl_response(self, response: str, width: int, height: int) -> Optional[List[float]]:
        """解析Qwen-VL响应"""
        try:
            # 清理JSON
            lines = response.splitlines()
            for i, line in enumerate(lines):
                if "```json" in line.lower():
                    json_output = "\n".join(lines[i+1:])
                    json_output = json_output.split("```")[0]
                    response = json_output.strip()
                    break
            
            # 解析JSON
            try:
                json_data = json.loads(response)
            except:
                try:
                    json_data = ast.literal_eval(response)
                except:
                    return None
            
            if not json_data or len(json_data) == 0:
                return None
            
            # 提取bbox
            first_item = json_data[0]
            if "bbox_2d" not in first_item:
                return None
            
            bbox_2d = first_item["bbox_2d"]
            if len(bbox_2d) != 4:
                return None
            
            x1, y1, x2, y2 = bbox_2d
            
            # 确保坐标顺序正确
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # 转换为[x, y, w, h]格式
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            
            if w <= 0 or h <= 0:
                return None
            
            return [x, y, w, h]
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

    def fuse_boxes(self, tracking_box: List[float], grounding_box: List[float],
                   tracking_conf: float) -> List[float]:
        """融合tracking和grounding结果"""
        # 计算IoU
        iou = self._calculate_iou(tracking_box, grounding_box)
        
        # 记录历史
        self.grounding_history.append({
            'iou': iou,
            'tracking_conf': tracking_conf,
        })
        
        # 低IoU时直接使用grounding结果
        if iou < self.min_iou_threshold:
            print(f"Low IoU ({iou:.3f}), using grounding result")
            return grounding_box
        
        # 动态权重融合
        confidence_factor = 1 - tracking_conf
        dynamic_weight = self.fusion_weight * confidence_factor
        
        fused_box = [
            (1 - dynamic_weight) * tracking_box[i] + dynamic_weight * grounding_box[i]
            for i in range(4)
        ]
        
        print(f"Fused: IoU={iou:.3f}, conf={tracking_conf:.3f}, weight={dynamic_weight:.3f}")
        return fused_box

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算IoU"""
        try:
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # 计算交集
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
            
        except:
            return 0.0

    def analyze_performance(self) -> Dict:
        """分析性能"""
        if not self.grounding_history:
            return {}
        
        ious = [h['iou'] for h in self.grounding_history]
        confs = [h['tracking_conf'] for h in self.grounding_history]
        
        return {
            'total_groundings': len(self.grounding_history),
            'avg_iou': np.mean(ious),
            'min_iou': np.min(ious),
            'max_iou': np.max(ious),
            'avg_conf_when_grounded': np.mean(confs),
        }