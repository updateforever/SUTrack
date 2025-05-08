#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step5_verification.py · 反向验证阶段
验证 step4 生成的多级描述是否能精准引导模型定位目标
支持多级别组合验证: 12 / 123 / 1234
"""

import os, sys, json, argparse, re
from typing import List, Dict, Optional
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageColor
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# 添加工程路径，加载数据集
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from lib.test.evaluation import get_dataset   # LaSOT loader

# ---------------- 常量 ----------------
IOU_THRESH = 0.4

# ----------- JSON / 文本 解析工具 ------------
def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(path: str, recs: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def compute_iou(b1, b2) -> float:
    x1,y1,x2,y2 = b1; x1g,y1g,x2g,y2g = b2
    xi1, yi1 = max(x1,x1g), max(y1,y1g)
    xi2, yi2 = min(x2,x2g), min(y2,y2g)
    inter = max(0, xi2-xi1)*max(0, yi2-yi1)
    a1 = (x2-x1)*(y2-y1); a2 = (x2g-x1g)*(y2g-y1g)
    return inter / (a1 + a2 - inter + 1e-6)

def parse_vlm_output(text: str) -> Dict[str,str]:
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    js = (m.group(1) if m else text).strip()
    try:
        return json.loads(js)
    except:
        return {}

def parse_detection_output(text: str) -> Optional[List[int]]:
    m = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text)
    js = m.group(1) if m else text.strip()
    if not js.endswith("]"): js += "]"
    try:
        arr = json.loads(js)
        if isinstance(arr, list) and arr:
            first = arr[0]
            bbox = first.get("bbox_2d") or first.get("bbox")
            if isinstance(bbox, list) and len(bbox)==4:
                return [int(v) for v in bbox]
    except:
        pass
    return None

# ----------- 模型 & 推理 ------------
def load_local_model(model_dir: str, device="cuda"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, device_map="auto", torch_dtype="auto",
        trust_remote_code=True
    ).eval()
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, repo_type="local_folder")
    return model, proc

@torch.no_grad()
def batch_infer_local(
    img_paths: List[str], prompts: List[str],
    model, proc, device="cuda", max_new=128
) -> List[str]:
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    chats = []
    for im, prm in zip(imgs, prompts):
        chats.append(proc.apply_chat_template(
            [{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":prm}]}],
            tokenize=False, add_generation_prompt=True
        ))
    inputs = proc(text=chats, images=imgs, return_tensors="pt", padding=True).to('cuda')
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    L = inputs["input_ids"].shape[1]
    return proc.batch_decode(out[:,L:], skip_special_tokens=True)

def api_infer(img_path: str, prompt: str, api_key: str) -> str:
    from openai import OpenAI
    import base64
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    with open(img_path,"rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    res = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[{"role":"user","content":[
            {{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}},
            {"type":"text","text":prompt}
        ]}]
    )
    return res.choices[0].message.content

# ------------ Prompt 构造 ------------
def build_verification_prompt(vlm_out: Dict[str,str], mode: str) -> str:
    parts = []
    if "1" in mode: parts += [vlm_out.get("level1","")]
    if "2" in mode: parts += [vlm_out.get("level2","")]
    if "3" in mode: parts += [vlm_out.get("level3","")]
    if "4" in mode: parts += [vlm_out.get("level4","")]
    desc = " ".join([p for p in parts if p]).strip()
    return f"""Locate the target object described below and output its bbox coordinates in JSON format.

Description:
{desc}
"""

# ----------- 验证流程 ------------
def verify_sequence(
    jsonl_path: str, seq, device, save_path: str,
    model=None, proc=None, api_key=None, levels=("12","123","1234")
):
    recs = load_jsonl(jsonl_path)
    out = []
    for r in tqdm(recs, desc=os.path.basename(jsonl_path)):
        # 原图路径
        img_path = seq.frames[r["frame_idx"]]
        gt       = r["gt_box"]
        vlm      = parse_vlm_output(r["vlm_output"])

        # 保存原图路径
        r["image_path"] = img_path

        for lvl in levels:
            prm = build_verification_prompt(vlm, lvl)
            if api_key:
                pred_txt = api_infer(img_path, prm, api_key)
                preds    = [pred_txt]
            else:
                preds = batch_infer_local([img_path], [prm], model, proc, device)

            pb = parse_detection_output(preds[0])
            ok = (compute_iou(gt, pb) >= IOU_THRESH) if pb else False

            r[f"ok_{lvl}"]   = ok
            r[f"pred_{lvl}"] = pb
            r[f"out_{lvl}"]  = preds[0]

        out.append(r)

    save_jsonl(save_path, out)
    print(f"✅ Saved: {save_path}")


# ----------------- CLI -----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  default="/home/wyp/project/SUTrack/soi/step4_vlt_outputs",
                   help="step4 输出目录")
    p.add_argument("--save_dir",   default="/home/wyp/project/SUTrack/soi/step5_verification_outputs",
                   help="step5 保存目录")
    p.add_argument("--dataset_name", default="lasot",
                   help="数据集名称")
    p.add_argument("--model_dir",  default="/mnt/first/wangyipei/qwenvl32b/",
                   help="本地模型路径（repo_type=local_folder）")
    p.add_argument("--api_key",    default=None,
                   help="DashScope API Key，若设置则走 API 模式")
    p.add_argument("--device",     default="auto",
                   help="本地推理设备")
    p.add_argument("--levels",     default="12,123,1234",
                   help="逗号分隔验证级别")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    ds = get_dataset(args.dataset_name)
    ds_map = {s.name:s for s in ds}

    model, proc = None, None
    if not args.api_key:
        model, proc = load_local_model(args.model_dir, device=args.device)

    levels = args.levels.split(",")
    for f in os.listdir(args.input_dir):
        if not f.endswith("_vlt.jsonl"): continue
        seq = f.replace("_vlt.jsonl","")
        if seq not in ds_map:
            print(f"⚠️ skip {seq}"); continue
        in_path  = os.path.join(args.input_dir, f)
        out_path = os.path.join(args.save_dir, seq+"_verify.jsonl")
        if os.path.exists(out_path):
            print(f"⚡️ exists, skip {seq}"); continue
        verify_sequence(
            in_path, ds_map[seq], args.device,
            out_path, model, proc, args.api_key, levels
        )

if __name__=="__main__":
    main()





