
import os
import json
import pandas as pd
import cv2


def stat_soi_frame_counts(soi_dir: str):
    stats = []

    for file in os.listdir(soi_dir):
        if not file.endswith("_soi_frames.jsonl"):
            continue
        path = os.path.join(soi_dir, file)
        with open(path, "r") as f:
            frame_indices = json.load(f)
        # print(file, len(frame_indices))
        stats.append({
            "sequence": file.replace("_soi_frames.jsonl", ""),
            "num_soi_frames": len(frame_indices),
            "min_index": min(frame_indices) if frame_indices else None,
            "max_index": max(frame_indices) if frame_indices else None
        })

    df = pd.DataFrame(stats).sort_values(by="num_soi_frames", ascending=False)
    print("📊 总序列数：", len(df))
    print("📊 平均干扰帧数：", round(df['num_soi_frames'].mean(), 2))
    print("📊 最大值：", df['num_soi_frames'].max())
    print("📊 最小值：", df['num_soi_frames'].min())

    print("\n📌 干扰帧数 Top 10 序列：")
    print(df.head(10))

    return df


def visualize_soi_sequence(
    seq_name: str,
    img_root: str,
    frame_jsonl: str,
    box_jsonl: str,
    save_dir: str = "./vis_soi"
):
    os.makedirs(save_dir, exist_ok=True)

    with open(frame_jsonl, "r") as f:
        soi_indices = json.load(f)

    with open(box_jsonl, "r") as f:
        box_lines = [json.loads(line.strip()) for line in f.readlines()]

    img_dir = os.path.join(img_root, seq_name.split("-")[0], seq_name, "img")

    for idx in soi_indices:
        frame_name = f"{idx+1:08d}.jpg"
        img_path = os.path.join(img_dir, frame_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ 图像读取失败: {img_path}")
            continue

        boxes = box_lines[idx]
        if len(boxes) == 0:
            continue
        
        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for box in boxes[1:]:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        out_path = os.path.join(save_dir, f"{seq_name}_{idx:04d}.jpg")
        cv2.imwrite(out_path, image)

    print(f"✅ 可视化完成，已保存至 {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["stat", "vis"], required=True)
    parser.add_argument("--soi_dir", type=str, help="用于统计的 SOI 帧路径")
    parser.add_argument("--seq_name", type=str, help="可视化的序列名称")
    parser.add_argument("--img_root", type=str, help="LaSOT 根路径")
    parser.add_argument("--frame_jsonl", type=str, help="SOI帧索引jsonl路径")
    parser.add_argument("--box_jsonl", type=str, help="步骤3的框jsonl路径")
    parser.add_argument("--save_dir", type=str, default="./vis_soi")

    args = parser.parse_args()

    if args.mode == "stat":
        if not args.soi_dir:
            print("❌ 缺少 --soi_dir 参数")
        else:
            stat_soi_frame_counts(args.soi_dir)

    elif args.mode == "vis":
        required = [args.seq_name, args.img_root, args.frame_jsonl, args.box_jsonl]
        if any(v is None for v in required):
            print("❌ 可视化缺少必要参数")
        else:
            visualize_soi_sequence(
                seq_name=args.seq_name,
                img_root=args.img_root,
                frame_jsonl=args.frame_jsonl,
                box_jsonl=args.box_jsonl,
                save_dir=args.save_dir
            )


# python soi/vis_and_analysis_step3_soi_results.py --mode stat --soi_dir /home/wyp/project/SUTrack/soi_outputs/mgit/step3_2_soi_frames_full
# /home/wyp/project/SUTrack/soi/step3_2_results1111111111111
# /home/wyp/project/SUTrack/soi/step3_soi_frames
            
"""
python soi/vis_and_analysis_step3_soi_results.py \
  --mode vis \
  --seq_name airplane-15 \
  --img_root /mnt/first/hushiyu/SOT/LaSOT/data \
  --frame_jsonl /home/wyp/project/SUTrack/soi/step3_2_results/airplane-15_soi_frames.jsonl \
  --box_jsonl /home/wyp/project/SUTrack/soi/step3_1_results/airplane-15.jsonl \
  --save_dir /home/wyp/project/SUTrack/soi/step3_vis/vis_airplane-15

"""