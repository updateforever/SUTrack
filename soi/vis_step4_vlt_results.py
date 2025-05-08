import json
import os
import cv2
import argparse

def draw_boxes(image, gt, candidates, texts):
    """绘制GT和干扰框及文本"""
    img = image.copy()
    cv2.rectangle(img, (gt['x1'], gt['y1']), (gt['x2'], gt['y2']), (0, 255, 0), 2)
    cv2.putText(img, "GT", (gt['x1'], gt['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i, box in enumerate(candidates):
        cv2.rectangle(img, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 0, 255), 2)
        cv2.putText(img, f"C{i+1}", (box['x1'], box['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    y0 = 30
    for level, text in texts.items():
        cv2.putText(img, f"{level}: {text[:80]}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y0 += 20

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="VLT标注jsonl文件")
    parser.add_argument("--save_dir", type=str, default="./vis_results", help="可视化结果保存目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.input_jsonl, 'r') as f:
        lines = f.readlines()

    for entry in lines:
        data = json.loads(entry)
        img_path = data['image']
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        gt = data['gt_box']
        candidates = data['candidates']
        descs = data['vlm_output']

        vis_img = draw_boxes(img, gt, candidates, descs)
        seq_name = os.path.basename(args.input_jsonl).replace('_vlt.jsonl', '')
        frame_id = data['frame_idx']
        save_path = os.path.join(args.save_dir, f"{seq_name}_{frame_id}.jpg")
        cv2.imwrite(save_path, vis_img)

    print("✅ 可视化完成")

if __name__ == "__main__":
    main()


# # 可视化结果查看
# python visualize_vlt_output.py --input_jsonl ./vlt_outputs/airplane-1_vlt.jsonl