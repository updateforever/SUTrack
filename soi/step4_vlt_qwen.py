import os
import sys
import json
import base64
import argparse
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI
from PIL import Image
import cv2

# 加入工程路径，确保能导入 get_dataset
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.test.evaluation import get_dataset  # 你已有的数据集加载函数

from configs.prompt_templates import build_prompt_multi, build_prompt, build_prompt_multi_cn, build_prompt_cn, build_structured_prompt

def draw_boxes_and_save_structured(
    image_path: str,
    gt_box: List[int],
    candidate_boxes: List[List[int]],
    save_root: str,
    frame_idx: int,
    category: str,
    seq_name: str
) -> str:
    """
    在图像上绘制 GT + 干扰框，并保存为结构化路径：
    save_root / category / seq_name / 000001.jpg
    """
    save_dir = os.path.join(save_root, category, seq_name)
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    # 绘制绿色 GT 框
    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
    # 绘制红色干扰框
    for box in candidate_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    filename = f"{frame_idx:06d}.jpg"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, img)
    return save_path

def load_soi_frame_indices(path: str) -> List[int]:
    """从包含帧号数组的一行 JSON 文件中读取索引列表"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_candidate_boxes(jsonl_path: str, frame_idx: int) -> List[Dict]:
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        if 0 <= frame_idx < len(lines):
            return json.loads(lines[frame_idx-1])  # 索引问题 TODO
        return []

# 提取类别信息
def extract_category_from_path(image_path: str) -> str:
    # 假设路径格式：/path/to/dataset/airplane/airplane-1/img/00000001.jpg
    parts = image_path.split(os.sep)
    for i in range(len(parts) - 1):
        if parts[i + 1].startswith(parts[i] + "-"):
            return parts[i]  # 例如返回 "airplane"
    return ""

def infer_with_qwen_stream(image_path: str, prompt: str, api_key: str) -> str:
    """调用 DashScope qwen 多模态模型（流式）"""
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
        b64_url = f"data:image/jpeg;base64,{b64}"

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",  #"qvq-max",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": b64_url}},
                {"type": "text", "text": prompt}
            ]
        }],
        # stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        # stream_options={
        #     "include_usage": True
        # }
    )
    answer_content = completion.choices[0].message.content
    print(completion.choices[0].message.content)
    # print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
    # for chunk in completion:
    #     # 如果chunk.choices为空，则打印usage
    #     if not chunk.choices:
    #         print("\nUsage:")
    #         print(chunk.usage)
    #     else:
    #         delta = chunk.choices[0].delta
    #         # 打印思考过程
    #         if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
    #             print(delta.reasoning_content, end='', flush=True)
    #             reasoning_content += delta.reasoning_content
    #         else:
    #             # 开始回复
    #             if delta.content != "" and is_answering is False:
    #                 print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
    #                 is_answering = True
    #             # 打印回复过程
    #             print(delta.content, end='', flush=True)
    #             answer_content += delta.content


    return answer_content


def infer_with_local(image_path: str, prompt: str, model_id: str, device: str) -> str:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    pipe = pipeline(Tasks.multi_modal_conversation, model=model_id, device=device)
    result = pipe({'image': image_path, 'text': prompt})
    return result["text"] if isinstance(result, dict) else str(result)


def run_inference(image_path: str, prompt: str, use_api: bool, **kwargs) -> str:
    try:
        if use_api:
            return infer_with_qwen_stream(image_path, prompt, kwargs['api_key'])
        else:
            return infer_with_local(image_path, prompt, kwargs['model_id'], kwargs['device'])
    except Exception as e:
        return f"[推理失败]: {str(e)}"


def save_jsonl(path: str, data: List[dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def build_output_entry(frame_idx: int, image_path: str, gt, candidates, prompt_obj, output_obj, multi: bool) -> Dict:
    entry = {
        "frame_idx": frame_idx,
        "image": image_path,
        "gt_box": gt,
        "candidates": candidates,
    }
    if multi:
        entry["prompts"] = prompt_obj
        entry["vlm_output"] = output_obj
    else:
        entry["prompt"] = prompt_obj
        entry["vlm_output"] = output_obj
    return entry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lasot")
    parser.add_argument("--soi_root", type=str, default="/home/wyp/project/SUTrack/soi/step3_2_results")
    parser.add_argument("--filtered_root", type=str, default="/home/wyp/project/SUTrack/soi/step3_1_results")
    parser.add_argument("--save_dir", type=str, default="/home/wyp/project/SUTrack/soi/step4_vlt_outputs")
    parser.add_argument("--use_api", action="store_true", help="是否调用 DashScope API")
    parser.add_argument("--api_key", type=str, default="sk-61547e720ce8407aa44f4511051903b0")
    parser.add_argument("--model_id", type=str, default="damo/cv_qwen_vl_base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi_level", action="store_true")
    parser.add_argument("--debug", action="store_true", help="是否启用调试模式，保存标注可视化图像")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = get_dataset(args.dataset_name)
    dataset_map = {seq.name: seq for seq in dataset}

    for soi_file in os.listdir(args.soi_root):
        if not soi_file.endswith("_soi_frames.jsonl"):
            continue
        seq_name = soi_file.replace("_soi_frames.jsonl", "")
        if seq_name not in dataset_map:
            continue
        if seq_name != "pig-2":
            continue
        seq = dataset_map[seq_name]
        soi_path = os.path.join(args.soi_root, soi_file)
        filtered_path = os.path.join(args.filtered_root, f"{seq_name}.jsonl")
        save_path = os.path.join(args.save_dir, f"{seq_name}_vlt.jsonl")

        if not os.path.exists(filtered_path):
            continue

        frame_indices = load_soi_frame_indices(soi_path)
        seq_outputs, skip_count = [], 0

        for frame_idx in tqdm(frame_indices, desc=f"处理 {seq_name}"):
            if frame_idx >= len(seq.frames):
                skip_count += 1
                continue

            image_path = seq.frames[frame_idx]

            boxes = load_candidate_boxes(filtered_path, frame_idx)
            if not boxes or len(boxes) < 2:  # 单框跳过，说明是
                skip_count += 1
                continue

            gt_box = boxes[0]
            candidate_boxes = boxes[1:]
            category = extract_category_from_path(image_path)

            image_path = draw_boxes_and_save_structured(
                image_path=seq.frames[frame_idx],
                gt_box=gt_box,
                candidate_boxes=candidate_boxes,
                save_root=os.path.join(args.save_dir, "framed_images"),
                frame_idx=frame_idx,
                category=category,
                seq_name=seq_name
            )

            if args.multi_level:
                prompts = build_prompt_multi_cn(gt_box, candidate_boxes, category=category)
                outputs = {
                    level: run_inference(
                        image_path, prompt, args.use_api,
                        api_key=args.api_key,
                        model_id=args.model_id,
                        device=args.device
                    ) for level, prompt in prompts.items()
                }
                entry = build_output_entry(frame_idx, image_path, gt_box, candidate_boxes, prompts, outputs, True)
            else:
                # prompt = build_prompt(gt_box, candidate_boxes)
                prompt = build_structured_prompt(gt_box, candidate_boxes, category=category)  # 中文测试
                output = run_inference(
                    image_path, prompt['en'], args.use_api,
                    api_key=args.api_key,
                    model_id=args.model_id,
                    device=args.device
                )
                entry = build_output_entry(frame_idx, image_path, gt_box, candidate_boxes, prompt, output, False)

            seq_outputs.append(entry)

        save_jsonl(save_path, seq_outputs)
        print(f"✅ {seq_name} 完成，跳过 {skip_count} 帧，保存至 {save_path}")


if __name__ == "__main__":
    main()
