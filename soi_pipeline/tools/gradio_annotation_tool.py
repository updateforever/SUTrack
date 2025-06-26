# import os
# import json
# import time
# import numpy as np
# import gradio as gr
# from PIL import Image, ImageDraw, ImageFont
# from typing import Dict, List, Optional
# from datetime import datetime
# import sys

# from gradio_image_annotation import image_annotator

# def pil_to_numpy(img: Image.Image):
#     return np.array(img)

# def numpy_to_dict(img: np.ndarray) -> Dict:
#     return {'image': img, 'bboxes': []}

# def load_jsonl(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

# def save_jsonl(file_path, data):
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for item in data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')

# # ------------------ 等比例缩放核心工具 ------------------

# def get_scaled_size(orig_w, orig_h, max_w, max_h):
#     """给定原始w/h与目标最大w/h，返回等比例缩放后尺寸和scale系数"""
#     scale = min(max_w / orig_w, max_h / orig_h)
#     new_w = int(orig_w * scale)
#     new_h = int(orig_h * scale)
#     return new_w, new_h, scale

# def resize_and_letterbox(img, max_w, max_h, fill_color=(128,128,128)):
#     """保持比例缩放，剩余填充灰色，返回新图、scale、pad"""
#     orig_w, orig_h = img.size
#     new_w, new_h, scale = get_scaled_size(orig_w, orig_h, max_w, max_h)
#     img_resized = img.resize((new_w, new_h), Image.LANCZOS)
#     new_img = Image.new("RGB", (max_w, max_h), fill_color)
#     pad_x = (max_w - new_w) // 2
#     pad_y = (max_h - new_h) // 2
#     new_img.paste(img_resized, (pad_x, pad_y))
#     return new_img, scale, pad_x, pad_y

# class HumanAnnotationTool:
#     def __init__(self, experiment_file: str, output_dir: str, display_size=(480, 360)):
#         self.experiment_file = experiment_file
#         self.output_dir = output_dir
#         self.results_file = os.path.join(output_dir, "human_annotation_results.jsonl")
#         self.display_size = display_size  # (W, H)

#         self.experiment_data = load_jsonl(experiment_file)
#         self.current_index = 0
#         self.completed_ids = set()
#         self.load_existing_results()
#         self.session_start_time = time.time()

#         print(f"✅ Loaded {len(self.experiment_data)} samples.")
#         print(f"📊 Progress: {len(self.completed_ids)}/{len(self.experiment_data)} completed")

#     def load_existing_results(self):
#         if os.path.exists(self.results_file):
#             results = load_jsonl(self.results_file)
#             self.completed_ids = {r['experiment_id'] for r in results}
#             for i, s in enumerate(self.experiment_data):
#                 if s['experiment_id'] not in self.completed_ids:
#                     self.current_index = i
#                     break
#             else:
#                 self.current_index = len(self.experiment_data)

#     def get_current_sample(self) -> Optional[Dict]:
#         if self.current_index >= len(self.experiment_data):
#             return None
#         return self.experiment_data[self.current_index]

#     def prepare_template_image(self, sample: Dict) -> Image.Image:
#         """模板图等比例缩放加灰边，目标框同样映射显示"""
#         try:
#             img = Image.open(sample['template_image_path']).convert('RGB')
#             orig_w, orig_h = img.size
#             box = sample['template_box']
#             new_img, scale, pad_x, pad_y = resize_and_letterbox(img, *self.display_size)
#             # 框也缩放并加偏移
#             x, y, w, h = box
#             x1 = int(x * scale + pad_x)
#             y1 = int(y * scale + pad_y)
#             x2 = int((x + w) * scale + pad_x)
#             y2 = int((y + h) * scale + pad_y)
#             draw = ImageDraw.Draw(new_img)
#             draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
#             try:
#                 font = ImageFont.truetype("arial.ttf", 16)
#             except:
#                 font = ImageFont.load_default()
#             draw.text((x1, max(y1 - 20, 0)), "TARGET", fill='red', font=font)
#             return new_img
#         except Exception as e:
#             print(f"Error preparing template image: {e}")
#             return Image.new('RGB', self.display_size, color='gray')

#     def prepare_current_image(self, sample: Dict) -> Dict:
#         """当前帧图像缩放letterbox，输出画布图"""
#         try:
#             img = Image.open(sample['current_image_path']).convert('RGB')
#             new_img, scale, pad_x, pad_y = resize_and_letterbox(img, *self.display_size)
#             return numpy_to_dict(pil_to_numpy(new_img))
#         except Exception as e:
#             print(f"Error loading current image: {e}")
#             return numpy_to_dict(np.array(Image.new('RGB', self.display_size, color='gray')))

#     def format_descriptions(self, sample: Dict) -> str:
#         descriptions = []
#         desc_map = {
#             "description_level_1": "🎯 **Position/Location**",
#             "description_level_2": "👁️ **Appearance/Visual Features**",
#             "description_level_3": "🏃 **Motion/Behavior Dynamics**",
#             "description_level_4": "🌍 **Environmental Context & Distractors**"
#         }
#         for key, title in desc_map.items():
#             desc = sample.get(key, "").strip()
#             if desc:
#                 descriptions.append(f"{title}: {desc}")
#         return "\n\n".join(descriptions) if descriptions else "No semantic descriptions available."

#     def create_progress_info(self) -> str:
#         total = len(self.experiment_data)
#         done = len(self.completed_ids)
#         current = min(self.current_index + 1, total)
#         progress_bar = "█" * (done * 20 // total) + "░" * (20 - done * 20 // total)
#         return (
#             f"📊 **Progress**: {done}/{total} ({done/total*100:.1f}%)\n"
#             f"[{progress_bar}]\n\n"
#             f"⏱️ **Session Time**: {(time.time() - self.session_start_time)/60:.1f} minutes\n"
#             f"📋 **Current Sample**: {current}/{total}"
#         )

#     def save_annotation(self, sample: Dict, annotation_data: Dict):
#         result = {
#             **sample,
#             "human_results": annotation_data,
#             "annotation_timestamp": datetime.now().isoformat(),
#             "status": "completed"
#         }
#         with open(self.results_file, 'a', encoding='utf-8') as f:
#             f.write(json.dumps(result, ensure_ascii=False) + '\n')
#         self.completed_ids.add(sample['experiment_id'])
#         print(f"✅ Saved annotation for {sample['experiment_id']}")

#     def create_interface(self) -> gr.Blocks:
#         with gr.Blocks(title="Human Annotation Tool", theme=gr.themes.Soft()) as interface:
#             current_sample_state = gr.State(value=None)
#             annotation_start_time = gr.State(value=time.time())
#             progress_display = gr.Markdown(value=self.create_progress_info())

#             gr.Markdown("""# 🧠 Human-Machine Cognitive Difference Study
# **Instructions**: 
# - 左边：模板目标
# - 右边：请圈出同一目标
# """)
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     template_image = gr.Image(
#                         label="Template with Target Box",
#                         type="numpy",
#                         interactive=False,
#                         height=self.display_size[1], width=self.display_size[0]
#                     )
#                 with gr.Column(scale=1):
#                     current_image = image_annotator(
#                         label="Click and drag to draw bounding box"
#                     )

#             descriptions_display = gr.Markdown(value="")
#             with gr.Row():
#                 semantic_level = gr.Radio(
#                     choices=[
#                         ("1 - Visual Only (Template Image)", 1),
#                         ("2 - + Position/Location Info", 2),
#                         ("3 - + Appearance/Visual Features", 3),
#                         ("4 - + Motion/Behavior Dynamics", 4),
#                         ("5 - + Environmental Context", 5),
#                         ("6 - Cannot Determine Target", 6)
#                     ],
#                     label="🧠 Which information level helped you most?",
#                     value=1
#                 )
#                 confidence_rating = gr.Slider(
#                     minimum=1, maximum=5, step=1, value=3,
#                     label="🎯 Confidence Level (1=Very Low, 5=Very High)"
#                 )
#                 difficulty_rating = gr.Slider(
#                     minimum=1, maximum=5, step=1, value=3,
#                     label="⚡ Task Difficulty (1=Very Easy, 5=Very Hard)"
#                 )

#             comments_input = gr.Textbox(
#                 label="💭 Comments (Optional)",
#                 placeholder="Any observations about this case?", lines=2
#             )
#             with gr.Row():
#                 skip_button = gr.Button("⏭️ Skip This Sample", variant="secondary")
#                 submit_button = gr.Button("✅ Submit Annotation", variant="primary")
#                 next_sample_button = gr.Button("➡️ Load Next Sample", variant="secondary")
#             status_display = gr.Markdown(value="")

#             def load_sample(sample_idx=None):
#                 if sample_idx is not None:
#                     self.current_index = sample_idx
#                 sample = self.get_current_sample()
#                 if sample is None:
#                     return (
#                         None, None, "🎉 **All samples completed!**", self.create_progress_info(),
#                         "✅ Experiment completed!", None, time.time()
#                     )
#                 template_img = self.prepare_template_image(sample)
#                 current_img = self.prepare_current_image(sample)
#                 descriptions = self.format_descriptions(sample)
#                 progress = self.create_progress_info()
#                 status = f"📋 **Sample {self.current_index + 1}**: {sample.get('sequence_name', '')} - Frame {sample.get('frame_idx', '')}"
#                 return (
#                     pil_to_numpy(template_img), current_img, descriptions,
#                     progress, status, sample, time.time()
#                 )

#             def submit_annotation(semantic_level, confidence, difficulty, comments,
#                                   sample, start_time, bbox_data):
#                 if sample is None:
#                     return "❌ No sample to annotate"
#                 if not bbox_data or len(bbox_data.get('bboxes', [])) == 0:
#                     return "❌ Please draw a bounding box on the current image"
#                 # === 坐标映射回原图 ===
#                 img = Image.open(sample['current_image_path']).convert('RGB')
#                 orig_w, orig_h = img.size
#                 canvas_w, canvas_h = self.display_size
#                 new_w, new_h, scale = get_scaled_size(orig_w, orig_h, canvas_w, canvas_h)
#                 pad_x = (canvas_w - new_w) // 2
#                 pad_y = (canvas_h - new_h) // 2

#                 bbox = bbox_data['bboxes'][0]
#                 x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
#                 # 去掉padding、除以scale，变回原图坐标
#                 x0_img = max((x0 - pad_x) / scale, 0)
#                 y0_img = max((y0 - pad_y) / scale, 0)
#                 x1_img = min((x1 - pad_x) / scale, orig_w)
#                 y1_img = min((y1 - pad_y) / scale, orig_h)
#                 box_canvas = [float(x0), float(y0), float(x1), float(y1)]
#                 box_original = [float(x0_img), float(y0_img), float(x1_img), float(y1_img)]

#                 annotation_data = {
#                     "selected_level": semantic_level,
#                     "bbox_canvas": box_canvas,
#                     "bbox_original": box_original,
#                     "confidence": confidence,
#                     "difficulty": difficulty,
#                     "comments": comments,
#                     "canvas_size": [canvas_w, canvas_h],
#                     "original_size": [orig_w, orig_h],
#                     "time_spent": time.time() - start_time
#                 }
#                 self.save_annotation(sample, annotation_data)
#                 self.current_index += 1
#                 return f"✅ Annotation saved! Sample {sample['experiment_id']} completed."

#             def skip_sample(sample):
#                 if sample is None:
#                     return "❌ No sample to skip"
#                 skip_data = {
#                     "selected_level": None,
#                     "bbox_canvas": None,
#                     "bbox_original": None,
#                     "confidence": None,
#                     "difficulty": None,
#                     "comments": "SKIPPED",
#                     "canvas_size": self.display_size,
#                     "original_size": None,
#                     "time_spent": 0
#                 }
#                 result = {
#                     **sample,
#                     "human_results": skip_data,
#                     "annotation_timestamp": datetime.now().isoformat(),
#                     "status": "skipped"
#                 }
#                 with open(self.results_file, 'a', encoding='utf-8') as f:
#                     f.write(json.dumps(result, ensure_ascii=False) + '\n')
#                 self.completed_ids.add(sample['experiment_id'])
#                 self.current_index += 1
#                 return f"⏭️ Sample {sample['experiment_id']} skipped."

#             # ---- Gradio事件绑定 ----
#             interface.load(
#                 fn=load_sample,
#                 inputs=[],
#                 outputs=[template_image, current_image, descriptions_display,
#                          progress_display, status_display, current_sample_state, annotation_start_time]
#             )
#             submit_button.click(
#                 fn=submit_annotation,
#                 inputs=[semantic_level, confidence_rating, difficulty_rating, comments_input,
#                         current_sample_state, annotation_start_time, current_image],
#                 outputs=[status_display]
#             ).then(
#                 fn=load_sample,
#                 inputs=[],
#                 outputs=[template_image, current_image, descriptions_display,
#                          progress_display, status_display, current_sample_state, annotation_start_time]
#             )
#             skip_button.click(
#                 fn=skip_sample,
#                 inputs=[current_sample_state],
#                 outputs=[status_display]
#             ).then(
#                 fn=load_sample,
#                 inputs=[],
#                 outputs=[template_image, current_image, descriptions_display,
#                          progress_display, status_display, current_sample_state, annotation_start_time]
#             )
#             # “下一样本”直接递增current_index并刷新
#             def load_next_sample(sample, *args):
#                 if sample is not None:
#                     self.current_index += 1
#                 return load_sample()

#             next_sample_button.click(
#                 fn=load_next_sample,
#                 inputs=[current_sample_state, template_image, current_image, descriptions_display,
#                         progress_display, status_display, current_sample_state, annotation_start_time],
#                 outputs=[template_image, current_image, descriptions_display,
#                          progress_display, status_display, current_sample_state, annotation_start_time]
#             )

#         return interface

#     def launch(self, **kwargs):
#         interface = self.create_interface()
#         print("🚀 Launching Human Annotation Tool...")
#         print(f"📁 Results: {self.results_file}")
#         launch_config = {
#             "share": False,
#             "server_name": "0.0.0.0",
#             "server_port": 7860,
#             "show_error": True,
#             **kwargs
#         }
#         interface.launch(**launch_config)

# def launch_annotation_tool(experiment_file: str, output_dir: str, **kwargs):
#     if not os.path.exists(experiment_file):
#         raise FileNotFoundError(f"Experiment file not found: {experiment_file}")
#     os.makedirs(output_dir, exist_ok=True)
#     tool = HumanAnnotationTool(experiment_file, output_dir)
#     tool.launch(**kwargs)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Human Annotation Tool")
#     parser.add_argument("--experiment-file", required=True, help="Path to experiment data file")
#     parser.add_argument("--output-dir", required=True, help="Output directory for results")
#     parser.add_argument("--port", type=int, default=7860, help="Server port")
#     parser.add_argument("--share", action="store_true", help="Create shareable link")
#     args = parser.parse_args()
#     launch_annotation_tool(
#         experiment_file=args.experiment_file,
#         output_dir=args.output_dir,
#         server_port=args.port,
#         share=args.share
#     )


import os
import json
import time
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

from gradio_image_annotation import image_annotator

def pil_to_numpy(img: Image.Image):
    return np.array(img)

def numpy_to_dict(img: np.ndarray) -> Dict:
    return {'image': img, 'bboxes': []}

def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    return data

def save_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ------------------ Aspect-preserving scaling utilities ------------------

def get_scaled_size(orig_w, orig_h, max_w, max_h):
    """Calculate scaled dimensions while preserving aspect ratio"""
    scale = min(max_w / orig_w, max_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    return new_w, new_h, scale

def resize_and_letterbox(img, max_w, max_h, fill_color=(128,128,128)):
    """Scale image preserving aspect ratio and add padding"""
    orig_w, orig_h = img.size
    new_w, new_h, scale = get_scaled_size(orig_w, orig_h, max_w, max_h)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new("RGB", (max_w, max_h), fill_color)
    pad_x = (max_w - new_w) // 2
    pad_y = (max_h - new_h) // 2
    new_img.paste(img_resized, (pad_x, pad_y))
    return new_img, scale, pad_x, pad_y

class HumanAnnotationTool:
    def __init__(self, experiment_file: str, output_dir: str, display_size=(640, 480)):
        self.experiment_file = experiment_file
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, "human_annotation_results.jsonl")
        self.display_size = display_size  # (W, H)

        self.experiment_data = load_jsonl(experiment_file)
        self.current_index = 0
        self.completed_ids = set()
        self.results_cache = []
        self.load_existing_results()
        self.session_start_time = time.time()

        print(f"✅ Loaded {len(self.experiment_data)} samples.")
        print(f"📊 Progress: {len(self.completed_ids)}/{len(self.experiment_data)} completed")

    def load_existing_results(self):
        if os.path.exists(self.results_file):
            self.results_cache = load_jsonl(self.results_file)
            self.completed_ids = {r['experiment_id'] for r in self.results_cache}
            # Find the first uncompleted sample
            for i, s in enumerate(self.experiment_data):
                if s['experiment_id'] not in self.completed_ids:
                    self.current_index = i
                    break
            else:
                self.current_index = len(self.experiment_data)

    def get_current_sample(self) -> Optional[Dict]:
        if 0 <= self.current_index < len(self.experiment_data):
            return self.experiment_data[self.current_index]
        return None

    def prepare_template_image(self, sample: Dict) -> Image.Image:
        """Scale template image with aspect ratio preservation and display target box"""
        try:
            img = Image.open(sample['template_image_path']).convert('RGB')
            orig_w, orig_h = img.size
            box = sample['template_box']
            new_img, scale, pad_x, pad_y = resize_and_letterbox(img, *self.display_size)
            
            # Scale and offset the bounding box
            x, y, w, h = box
            x1 = int(x * scale + pad_x)
            y1 = int(y * scale + pad_y)
            x2 = int((x + w) * scale + pad_x)
            y2 = int((y + h) * scale + pad_y)
            
            draw = ImageDraw.Draw(new_img)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            draw.text((x1, max(y1 - 20, 0)), "TARGET", fill='red', font=font)
            
            return new_img
        except Exception as e:
            print(f"Error preparing template image: {e}")
            return Image.new('RGB', self.display_size, color='gray')

    def prepare_current_image(self, sample: Dict) -> Dict:
        """Scale current frame with aspect ratio preservation"""
        try:
            img = Image.open(sample['current_image_path']).convert('RGB')
            new_img, scale, pad_x, pad_y = resize_and_letterbox(img, *self.display_size)
            return numpy_to_dict(pil_to_numpy(new_img))
        except Exception as e:
            print(f"Error loading current image: {e}")
            return numpy_to_dict(np.array(Image.new('RGB', self.display_size, color='gray')))

    def format_descriptions(self, sample: Dict) -> str:
        descriptions = []
        desc_map = {
            "description_level_1": "🎯 **Position/Location**",
            "description_level_2": "👁️ **Appearance/Visual Features**",
            "description_level_3": "🏃 **Motion/Behavior Dynamics**",
            "description_level_4": "🌍 **Environmental Context & Distractors**"
        }
        for key, title in desc_map.items():
            desc = sample.get(key, "").strip()
            if desc:
                descriptions.append(f"{title}: {desc}")
        return "\n\n".join(descriptions) if descriptions else "No semantic descriptions available."

    def create_progress_info(self) -> str:
        total = len(self.experiment_data)
        done = len(self.completed_ids)
        current = min(self.current_index + 1, total)
        progress_bar = "█" * (done * 20 // total) + "░" * (20 - done * 20 // total)
        return (
            f"📊 **Progress**: {done}/{total} ({done/total*100:.1f}%)\n"
            f"[{progress_bar}]\n\n"
            f"⏱️ **Session Time**: {(time.time() - self.session_start_time)/60:.1f} minutes\n"
            f"📋 **Current Sample**: {current}/{total}"
        )

    def save_annotation(self, sample: Dict, annotation_data: Dict) -> None:
        """Save annotation to results file and update cache"""
        result = {
            **sample,
            "human_results": annotation_data,
            "annotation_timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        self.results_cache.append(result)
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        self.completed_ids.add(sample['experiment_id'])
        print(f"✅ Saved annotation for {sample['experiment_id']}")

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Human Annotation Tool", theme=gr.themes.Soft()) as interface:
            current_sample_state = gr.State(value=None)
            annotation_start_time = gr.State(value=time.time())
            
            gr.Markdown("""# 🧠 Human-Machine Cognitive Difference Study
            
## Instructions
- **Left**: Template image with target object highlighted in red
- **Right**: Draw a bounding box around the same target in the current frame
- Use semantic descriptions if needed to help locate the target
- Rate your confidence and the task difficulty
""")
            
            progress_display = gr.Markdown(value=self.create_progress_info())
            
            with gr.Row():
                with gr.Column(scale=1):
                    template_image = gr.Image(
                        label="Template with Target Box",
                        type="numpy",
                        interactive=False,
                        height=self.display_size[1], 
                        width=self.display_size[0]
                    )
                    descriptions_display = gr.Markdown(
                        label="Semantic Descriptions",
                        value="",
                        elem_classes=["semantic-descriptions"]
                    )
                
                with gr.Column(scale=1):
                    current_image = image_annotator(
                        label="Draw a bounding box around the target",
                        height=self.display_size[1],
                        width=self.display_size[0],
                        # show_label=False,
                        disable_edit_boxes=True,
                    )
                    status_display = gr.Markdown(
                        value="",
                        elem_classes=["status-display"]
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    semantic_level = gr.Radio(
                        choices=[
                            ("1 - Visual Only (Template Image)", 1),
                            ("2 - + Position/Location Info", 2),
                            ("3 - + Appearance/Visual Features", 3),
                            ("4 - + Motion/Behavior Dynamics", 4),
                            ("5 - + Environmental Context", 5),
                            ("6 - Cannot Determine Target", 6)
                        ],
                        label="🧠 Which information level helped you most?",
                        value=1
                    )
                
                with gr.Column(scale=1):
                    with gr.Row():
                        confidence_rating = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="🎯 Confidence Level",
                            info="1=Very Low, 5=Very High"
                        )
                    
                    with gr.Row():
                        difficulty_rating = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="⚡ Task Difficulty",
                            info="1=Very Easy, 5=Very Hard"
                        )
            
            comments_input = gr.Textbox(
                label="💭 Comments (Optional)",
                placeholder="Any observations about this case?", 
                lines=2
            )
            
            with gr.Row():
                back_button = gr.Button("⬅️ Previous Sample", variant="secondary")
                skip_button = gr.Button("⏭️ Skip This Sample", variant="secondary")
                submit_button = gr.Button("✅ Submit Annotation", variant="primary")
                next_button = gr.Button("➡️ Next Sample", variant="secondary")

            # Custom CSS for better UI
            gr.HTML("""
            <style>
                .semantic-descriptions {
                    max-height: 200px;
                    overflow-y: auto;
                    padding: 10px;
                    background-color: #f5f5f5;
                    border-radius: 8px;
                    margin-top: 10px;
                }
                .status-display {
                    padding: 8px;
                    margin-top: 10px;
                    border-radius: 8px;
                    background-color: #e8f4f8;
                }
            </style>
            """)

            # ---- Load sample function ----
            def load_sample(sample_idx=None):
                if sample_idx is not None:
                    self.current_index = max(0, min(sample_idx, len(self.experiment_data) - 1))
                
                sample = self.get_current_sample()
                if sample is None:
                    return (
                        None, None, "🎉 **All samples completed!**", self.create_progress_info(),
                        "✅ Experiment completed!", None, time.time()
                    )
                
                template_img = self.prepare_template_image(sample)
                current_img = self.prepare_current_image(sample)
                descriptions = self.format_descriptions(sample)
                progress = self.create_progress_info()
                status = f"📋 **Sample {self.current_index + 1}**: {sample.get('sequence_name', '')} - Frame {sample.get('frame_idx', '')}"
                
                return (
                    pil_to_numpy(template_img), current_img, descriptions,
                    progress, status, sample, time.time()
                )
            
            # ---- Submit annotation function ----
            def submit_annotation(semantic_level, confidence, difficulty, comments,
                                sample, start_time, bbox_data):
                if sample is None:
                    return "❌ No sample to annotate"
                
                if not bbox_data or len(bbox_data.get('bboxes', [])) == 0:
                    return "❌ Please draw a bounding box on the current image"
                
                # Map coordinates back to original image
                img = Image.open(sample['current_image_path']).convert('RGB')
                orig_w, orig_h = img.size
                canvas_w, canvas_h = self.display_size
                new_w, new_h, scale = get_scaled_size(orig_w, orig_h, canvas_w, canvas_h)
                pad_x = (canvas_w - new_w) // 2
                pad_y = (canvas_h - new_h) // 2

                bbox = bbox_data['bboxes'][0]
                x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
                
                # Remove padding and scale back to original coordinates
                x0_img = max((x0 - pad_x) / scale, 0)
                y0_img = max((y0 - pad_y) / scale, 0)
                x1_img = min((x1 - pad_x) / scale, orig_w)
                y1_img = min((y1 - pad_y) / scale, orig_h)
                
                box_canvas = [float(x0), float(y0), float(x1), float(y1)]
                box_original = [float(x0_img), float(y0_img), float(x1_img), float(y1_img)]

                annotation_data = {
                    "selected_level": semantic_level,
                    "bbox_canvas": box_canvas,
                    "bbox_original": box_original,
                    "confidence": confidence,
                    "difficulty": difficulty,
                    "comments": comments,
                    "canvas_size": [canvas_w, canvas_h],
                    "original_size": [orig_w, orig_h],
                    "time_spent": time.time() - start_time
                }
                
                self.save_annotation(sample, annotation_data)
                self.current_index += 1
                
                # Return updated progress info along with success message
                return f"✅ Annotation saved! Sample {sample['experiment_id']} completed."
            
            # ---- Skip sample function ----
            def skip_sample(sample):
                if sample is None:
                    return "❌ No sample to skip"
                
                skip_data = {
                    "selected_level": None,
                    "bbox_canvas": None,
                    "bbox_original": None,
                    "confidence": None,
                    "difficulty": None,
                    "comments": "SKIPPED",
                    "canvas_size": self.display_size,
                    "original_size": None,
                    "time_spent": 0
                }
                
                result = {
                    **sample,
                    "human_results": skip_data,
                    "annotation_timestamp": datetime.now().isoformat(),
                    "status": "skipped"
                }
                
                self.results_cache.append(result)
                with open(self.results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                self.completed_ids.add(sample['experiment_id'])
                self.current_index += 1
                
                return f"⏭️ Sample {sample['experiment_id']} skipped."
            
            # ---- Go to previous sample function ----
            def go_to_previous_sample():
                if self.current_index > 0:
                    self.current_index -= 1
                    return "⬅️ Loaded previous sample"
                return "❌ Already at the first sample"
            
            # ---- Go to next sample function ----
            def go_to_next_sample():
                if self.current_index < len(self.experiment_data) - 1:
                    self.current_index += 1
                    return "➡️ Loaded next sample"
                return "❌ Already at the last sample"

            # ---- Event bindings ----
            # Initial load
            interface.load(
                fn=load_sample,
                inputs=[],
                outputs=[template_image, current_image, descriptions_display,
                        progress_display, status_display, current_sample_state, annotation_start_time]
            )
            
            # Submit button
            submit_button.click(
                fn=submit_annotation,
                inputs=[semantic_level, confidence_rating, difficulty_rating, comments_input,
                        current_sample_state, annotation_start_time, current_image],
                outputs=[status_display]
            ).then(
                fn=load_sample,
                inputs=[],
                outputs=[template_image, current_image, descriptions_display,
                        progress_display, status_display, current_sample_state, annotation_start_time]
            )
            
            # Skip button
            skip_button.click(
                fn=skip_sample,
                inputs=[current_sample_state],
                outputs=[status_display]
            ).then(
                fn=load_sample,
                inputs=[],
                outputs=[template_image, current_image, descriptions_display,
                        progress_display, status_display, current_sample_state, annotation_start_time]
            )
            
            # Back button
            back_button.click(
                fn=go_to_previous_sample,
                inputs=[],
                outputs=[status_display]
            ).then(
                fn=load_sample,
                inputs=[],
                outputs=[template_image, current_image, descriptions_display,
                        progress_display, status_display, current_sample_state, annotation_start_time]
            )
            
            # Next button
            next_button.click(
                fn=go_to_next_sample,
                inputs=[],
                outputs=[status_display]
            ).then(
                fn=load_sample,
                inputs=[],
                outputs=[template_image, current_image, descriptions_display,
                        progress_display, status_display, current_sample_state, annotation_start_time]
            )

        return interface

    def launch(self, **kwargs):
        interface = self.create_interface()
        print("🚀 Launching Human Annotation Tool...")
        print(f"📁 Results: {self.results_file}")
        launch_config = {
            "share": False,
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "show_error": True,
            **kwargs
        }
        interface.launch(**launch_config)

def launch_annotation_tool(experiment_file: str, output_dir: str, **kwargs):
    if not os.path.exists(experiment_file):
        raise FileNotFoundError(f"Experiment file not found: {experiment_file}")
    os.makedirs(output_dir, exist_ok=True)
    tool = HumanAnnotationTool(experiment_file, output_dir)
    tool.launch(**kwargs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Human Annotation Tool")
    parser.add_argument("--experiment-file", required=True, help="Path to experiment data file")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create shareable link")
    args = parser.parse_args()
    launch_annotation_tool(
        experiment_file=args.experiment_file,
        output_dir=args.output_dir,
        server_port=args.port,
        share=args.share
    )