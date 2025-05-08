import cv2 as cv
import numpy as np
import json
import os


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_filename(path):
    """返回排序后的图像路径列表"""
    filenames = []
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filenames.append(os.path.join(root, file))
    return sorted(filenames)


def draw_lasot_with_mask_and_status(seq_name):
    """绘制可视化图像，包括GT、预测框、mask和status状态文本"""
    # 路径设置
    base_dir = '/home/wyp/project/SUTrack/output/lasot_demo_with_status'
    image_path = '/mnt/first/hushiyu/SOT/LaSOT/data/%s/%s' % (seq_name.split('-')[0], seq_name)
    gt_path = '/mnt/first/hushiyu/SOT/LaSOT/data/%s/%s/groundtruth.txt' % (seq_name.split('-')[0], seq_name)
    pred_path = f'/home/wyp/project/SUTrack/output/test/soi_online_tracking_results/odtrack/baseline/lasot/{seq_name}.txt'
    status_path = f'{pred_path.replace(".txt", "_status.txt")}'
    mask_path = f'{pred_path.replace(".txt", "_mask.jsonl")}'

    save_seq_dir = os.path.join(base_dir, seq_name)
    makedir(save_seq_dir)

    filenames = read_filename(image_path)
    gts = np.loadtxt(gt_path, delimiter=',')
    preds = np.loadtxt(pred_path, delimiter='\t')
    
    # 读取 status 信息
    with open(status_path, 'r') as f:
        status_list = [line.strip() for line in f.readlines()]

    # 读取 mask 信息
    with open(mask_path, 'r') as f:
        raw_mask_lines = [json.loads(line) for line in f.readlines()]
    # 构造对齐后的 mask 列表，前两帧为空，后面每一帧超前读取2帧数据
    mask_lines = [[] for _ in range(2)] + raw_mask_lines
    # 若总帧数超过mask长度+2，补足空mask，避免越界
    while len(mask_lines) < len(filenames):
        mask_lines.append([])

    assert len(gts) == len(preds) == len(status_list)+1 == len(mask_lines)-1 == len(filenames)

    for i in range(len(filenames)):
        if i == 0:
            continue
        img = cv.imread(filenames[i])
        name = os.path.basename(filenames[i])
        gt = [int(x) for x in gts[i]]
        pred = [int(x) for x in preds[i]]
        status = status_list[i-1]
        masks = mask_lines[i]

        # 绘制 GT 框：绿色
        cv.rectangle(img, (gt[0], gt[1]), (gt[0] + gt[2], gt[1] + gt[3]), (0, 255, 0), 2)
        # 绘制预测框：蓝色
        cv.rectangle(img, (pred[0], pred[1]), (pred[0] + pred[2], pred[1] + pred[3]), (255, 0, 0), 2)
        if status != "absent":
            # 绘制 mask 区域：虚线红框
            for box in masks:
                x = int(box["x1"])
                y = int(box["y1"])
                w = int(box["x2"] - box["x1"])
                h = int(box["y2"] - box["y1"])
                # 然后绘制虚线框
                for j in range(x, x + w, 5):
                    cv.line(img, (j, y), (j + 2, y), (0, 0, 255), 1)
                    cv.line(img, (j, y + h), (j + 2, y + h), (0, 0, 255), 1)
                for j in range(y, y + h, 5):
                    cv.line(img, (x, j), (x, j + 2), (0, 0, 255), 1)
                    cv.line(img, (x + w, j), (x + w, j + 2), (0, 0, 255), 1)

        # 显示状态文字
        cv.putText(img, f"Status: {status}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 保存可视化图
        cv.imwrite(os.path.join(save_seq_dir, name), img)

    print(f"Done: {seq_name}")


if __name__ == '__main__':
    draw_lasot_with_mask_and_status("sheep-3")
