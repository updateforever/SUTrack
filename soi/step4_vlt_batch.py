#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step4_vlt_batch.py · Batch inference with local Qwen2.5-VL-32B
"""

import os, sys, json, argparse
from typing import List, Dict
from tqdm import tqdm
from PIL import Image
import cv2
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ────────── 工程路径 & 数据集加载 ──────────
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.test.evaluation import get_dataset                    # LaSOT loader

from configs.prompt_templates import build_structured_prompt   # 你的 prompt 构造函数

# ────────── 绘框 & 工具函数 ──────────
# ────────── Markdown 可视化 ──────────
def save_md_visualization(seq_records: List[Dict], md_root: str, seq_name: str, category: str):
    """
    生成 markdown，可视化每帧图片 + VLM 输出
    """
    if not seq_records:
        return

    md_dir = os.path.join(md_root, category, seq_name)
    os.makedirs(md_dir, exist_ok=True)
    md_path = os.path.join(md_dir, f"{seq_name}.md")

    with open(md_path, "w", encoding="utf-8") as md:
        md.write(f"# {seq_name} · VLT Results\n\n")
        for rec in seq_records:
            frame_id = rec["frame_idx"]
            # 1) 显示图片（相对路径）
            img_rel = os.path.relpath(rec["image"], md_dir)
            md.write(f"## Frame {frame_id}\n\n")
            md.write(f"![frame{frame_id}]({img_rel})\n\n")

            # 2) 解析 JSON 输出
            try:
                desc = json.loads(rec["vlm_output"])
                md.write(f"- **Location**: {desc.get('level1','')}\n")
                md.write(f"- **Appearance**: {desc.get('level2','')}\n")
                md.write(f"- **Dynamic**: {desc.get('level3','')}\n")
                md.write(f"- **Distractor**: {desc.get('level4','')}\n\n")
            except Exception:
                # 若非 JSON 格式，直接写文本
                md.write(rec["vlm_output"] + "\n\n")
    return md_path

def draw_boxes_and_save(image_path, gt, cands,
                        save_root, frame_idx, category, seq_name):
    save_dir = os.path.join(save_root, category, seq_name)
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(image_path)
    assert img is not None

    def get_thickness(box):
        """根据目标框的尺寸自适应线宽"""
        w = box[2] - box[0]
        h = box[3] - box[1]
        base = min(w, h)
        if base < 50:
            return 1
        elif base < 100:
            return 1
        else:
            return 2

    # 绘制 GT 框（绿色）
    thickness_gt = get_thickness(gt)
    cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), thickness_gt)

    # 绘制干扰框（红色）
    for b in cands:
        thickness = get_thickness(b)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness)

    out_path = os.path.join(save_dir, f"{frame_idx:06d}.jpg")
    cv2.imwrite(out_path, img)
    return out_path


def load_soi_indices(path):               # json list  [1,5,9...]
    with open(path,'r',encoding='utf-8') as f: return json.load(f)

def load_candidate_boxes(jsonl_path, idx):# idx 从 1 开始
    with open(jsonl_path) as f:
        lines = f.readlines()
        return json.loads(lines[idx-1]) if 0 <= idx-1 < len(lines) else []

def extract_category(p):                  # 数据集路径推类名
    parts = p.split(os.sep)
    for i in range(len(parts)-1):
        if parts[i+1].startswith(parts[i]+'-'): return parts[i]
    return ""

def save_jsonl(path, data):
    with open(path,'w',encoding='utf-8') as f:
        for d in data: f.write(json.dumps(d, ensure_ascii=False)+'\n')

def build_entry(idx,img,gt,cands,prompt,out):
    return dict(frame_idx=idx,image=img,gt_box=gt,candidates=cands,
                prompt=prompt,vlm_output=out)

# ────────── Qwen-VL 加载 & 批量推理 ──────────
def load_qwenvl(model_dir):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, 
        torch_dtype='auto', 
        device_map='auto',
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return model, processor

@torch.no_grad()
def batch_infer(img_paths: List[str], prompts: List[str],
                model, proc, max_new=192) -> List[str]:
    """一次推 batch_size 张图 + prompt"""
    images  = [Image.open(p).convert("RGB") for p in img_paths]
    # 1) 组装 messages → chat_text 列表
    chat_texts = []
    for img, prm in zip(images, prompts):
        msgs=[{"role":"user","content":[{"type":"image","image":img},
                                       {"type":"text","text":prm}]}]
        chat_texts.append(proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        )
    # 2) processor 打包
    inputs = proc(text=chat_texts, images=images,
                  return_tensors="pt", padding=True).to('cuda')
    # 3) 生成
    out = model.generate(**inputs, max_new_tokens=max_new,
                         do_sample=True, temperature=0.3)
    prompt_len = inputs["input_ids"].shape[1]
    return proc.batch_decode(out[:,prompt_len:], skip_special_tokens=True)

# ────────── CLI 参数 ──────────
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="lasot")
    ap.add_argument("--soi_root", default="/your/step3_2_results")
    ap.add_argument("--filtered_root", default="/your/step3_1_results")
    ap.add_argument("--save_dir", default="./step4_vlt_outputs")
    ap.add_argument("--model_dir", required=True, help="local dir of qwen2_5-vl-32b")
    ap.add_argument("--batch_size", type=int, default=1)
    return ap.parse_args()

# ────────── MAIN ──────────
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = get_dataset(args.dataset_name)
    seq_map = {s.name: s for s in dataset}
    model, proc = load_qwenvl(args.model_dir)

    for soi_file in os.listdir(args.soi_root):
        if not soi_file.endswith("_soi_frames.jsonl"):
            continue
        seq_name = soi_file.replace("_soi_frames.jsonl", "")
        if seq_name not in seq_map:
            continue

        soi_path = os.path.join(args.soi_root, soi_file)
        filt_path = os.path.join(args.filtered_root, f"{seq_name}.jsonl")
        if not os.path.exists(filt_path):
            continue

        frames = load_soi_indices(soi_path)
        seq_out, skip = [], 0
        b_imgs, b_prompts, b_meta = [], [], []

        # 🚨 检查是否已有 MD 文件，若有则跳过
        cat = extract_category(seq_map[seq_name].frames[0])
        md_dir = os.path.join(args.save_dir, "vis_md", cat, seq_name)
        md_path = os.path.join(md_dir, f"{seq_name}.md")
        if os.path.exists(md_path):
            print(f"⚠️ 检测到 {seq_name} 已完成，跳过。")
            continue

        for idx in tqdm(frames, desc=seq_name):
            if idx >= len(seq_map[seq_name].frames):
                skip += 1
                continue
            boxes = load_candidate_boxes(filt_path, idx)
            if len(boxes) < 2:
                skip += 1
                continue

            img_raw = seq_map[seq_name].frames[idx]
            img_vis = draw_boxes_and_save(img_raw, boxes[0], boxes[1:],
                                          os.path.join(args.save_dir, "framed_images"), idx, cat, seq_name)
            prompt = build_structured_prompt(boxes[0], boxes[1:], cat)['en']
            b_imgs.append(img_vis)
            b_prompts.append(prompt)
            b_meta.append((idx, img_vis, boxes[0], boxes[1:], prompt))

            if len(b_imgs) == args.batch_size or idx == frames[-1]:
                outs = batch_infer(b_imgs, b_prompts, model, proc)
                for m, o in zip(b_meta, outs):
                    seq_out.append(build_entry(*m, o))
                b_imgs, b_prompts, b_meta = [], [], []

        out_path = os.path.join(args.save_dir, f"{seq_name}_vlt.jsonl")
        save_jsonl(out_path, seq_out)

        md_root = os.path.join(args.save_dir, "vis_md")
        md_path = save_md_visualization(seq_out, md_root, seq_name, cat)

        print(f"✅ {seq_name} finished · skipped {skip} · "
              f"jsonl: {out_path} · md: {md_path}")


if __name__ == "__main__":
    main()

"""
python step4_vlt_batch.py \
  --model_dir /mnt/models/qwen2_5-vl-32b \
  --batch_size 4 \
  --fp16
/project/SUTrack$ python soi/step4_vlt_batch.py --model_dir /mnt/first/wangyipei/qwenvl32b/ --batch_size 1 --save_dir /home/wyp/project/SUTrack/soi/step4_vlt_outputs --filtered_root /home/wyp/project/SUTrack/soi/step3_1_results --soi_root /home/wyp/project/SUTrack/soi/step3_2_results
"""