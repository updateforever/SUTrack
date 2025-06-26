# pytracking/soi_pipeline/models/vlm_interface.py
import base64
import time
import json
from typing import Union, List, Tuple
from openai import OpenAI

# 尝试导入本地模型依赖
try:
    import torch
    from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    torch = None


class VLMEngine:
    """视觉语言模型推理引擎"""
    
    def __init__(self, config):
        self.config = config
        self.processor = None
        self.model = None
        self.client = None
        
        if config.use_local_model:
            if QWEN_AVAILABLE:
                self._init_local_model()
            else:
                print("Warning: Local model dependencies not available, switching to API mode")
                self.config.use_local_model = False
                self._init_api_client()
        else:
            self._init_api_client()
    
    def _init_local_model(self):
        """初始化本地Qwen模型"""
        if not self.config.model_path:
            raise ValueError("model_path must be specified for local model")
        
        print(f"🔄 Loading local model from {self.config.model_path}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path, 
            use_fast=True, 
            trust_remote_code=True
        )
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        print("✅ Local model loaded successfully")
    
    def _init_api_client(self):
        """初始化API客户端"""
        if not self.config.api_key:
            raise ValueError("api_key must be specified for API mode")
        
        self.client = OpenAI(api_key=self.config.api_key)
        print("✅ API client initialized")
    
    def _encode_image(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    
    def generate(self, image_paths: Union[str, List[str]], prompt: str) -> str:
        """生成VLM描述"""
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        start_time = time.time()
        
        try:
            if self.config.use_local_model and self.model:
                result, input_width, input_height = self._generate_local(image_paths, prompt)
            else:
                result = self._generate_api(image_paths, prompt)
                input_width, input_height = None, None  # TODO

            if self.config.verbose:
                elapsed = time.time() - start_time
                print(f"🕒 VLM inference completed in {elapsed:.2f}s")
            
            return result, input_width, input_height
        
        except Exception as e:
            print(f"❌ VLM generation failed: {e}")
            return ""
    
    def _generate_local(self, image_paths: List[str], prompt: str) -> str:
        """使用本地模型生成"""
        # 构建消息格式
        image_msgs = [{"type": "image", "image": path} for path in image_paths]
        messages = [{
            "role": "user", 
            "content": image_msgs + [{"type": "text", "text": prompt}]
        }]
        
        # 处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        input_height = inputs['image_grid_thw'][0][1] * 14
        input_width = inputs['image_grid_thw'][0][2] * 14

        # 移动到设备
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True
            )
        
        # 解码输出
        trimmed_ids = [
            output[len(input_ids):] 
            for input_ids, output in zip(inputs["input_ids"], output_ids)
        ]
        
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0], input_width, input_height
    
    def _generate_api(self, image_paths: List[str], prompt: str) -> str:
        """使用API生成"""
        # 编码图像
        images = [
            {"type": "image_url", "image_url": {"url": self._encode_image(path)}} 
            for path in image_paths
        ]
        
        messages = [{
            "role": "user", 
            "content": images + [{"type": "text", "text": prompt}]
        }]
        
        # API调用
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content


def build_structured_prompt(gt_box: List[float], candidate_boxes: List[List[float]],
                            category: str = "", template_mode: bool = False) -> str:
    """构建认知语言学引导下的结构化提示词（适配目标跟踪任务）"""

    category_text = f"The target is a {category}." if category else "The target object is clearly marked."

    if template_mode:
        scene_description = f"""
You will be given two images:
- The **first** image is a template frame that clearly shows the tracking target.
- The **second** image is the current frame, where the same target appears among several visually similar distractors.

{category_text}
"""
    else:
        scene_description = f"""
You are observing an image that contains the tracking target, marked clearly by a green bounding box. 
There are also several visually similar distractor objects in the scene.

{category_text}
"""

    instruction = f"""
⚠️  The green bounding boxes in the image(s) are provided only to help you analyze the scene.
**Do NOT mention any bounding boxes, coordinates, or technical annotation terms in your description.**

Your task is to produce a concise, structured, multi-level semantic description of the tracking target,
guided strictly by two principles from cognitive linguistics: 
• **Concretization** (vivid, specific, and easily imaginable details)  
• **Saliency guiding** (highlighting distinctive features that rapidly differentiate the target from distractors)

Return your answer strictly in this JSON format:
{{ 
"level1": "<Location Feature>", "level2": "<Appearance Description>", "level3": "<Dynamic State Description>", "level4": "<Distractor Differentiation>"
}}

Instructions for Description Generation
--------------------------
- **Location Feature**
  • Start with a preposition and end with a comma.
  • Describe the semantic location (e.g., “At the center of the roadway,”).
  • Never include coordinates or annotation terms.

- **Appearance Description**
  • Use one of these formats:
    1️⃣ “a/an [adjective(s)] [object]”
    2️⃣ “a/an [adjective(s)] [object] on/in [carrier]”
    3️⃣ “a/an [adjective(s)] [object] held/carried by [carrier]”
  • Always include **color + object type**, plus salient visual features (size, texture, shape, etc.)

- **Dynamic State Description**
  • Output a **complete verb phrase** that continues the sentence.
  • Describe the motion/pose/state of the target or its carrier (e.g., “is running along the sidewalk”)

- **Distractor Differentiation**
  • Start each phrase with “to the target’s [direction]”
  • Based on the scene, **autonomously identify any elements** that may confuse a tracker.  
  • Use clear directional and visual distinctions
  • Avoid vague terms like “different” or “unlike”
  • If the scene is simple with no strong distractors, you may briefly describe nearby semantic objects instead or leave this field concise.

Output only the JSON object, without any explanation or markdown syntax.
"""

    full_prompt = scene_description.strip() + "\n" + instruction.strip()
    return full_prompt



