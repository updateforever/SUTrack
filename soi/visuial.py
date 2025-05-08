import os
import cv2 as cv
import numpy as np


def makedir(path):
    """根据指定路径创建文件夹"""
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def read_filenames(path, extension='.jpg'):
    """返回指定路径下的文件名称"""
    return sorted(
        os.path.join(root, file)
        for root, _, files in os.walk(path)
        for file in files
        if file.endswith(extension)
    )


def load_data(file_path, delimiter=','):
    """加载标注数据"""
    data = np.loadtxt(file_path, delimiter=delimiter)
    return np.where(np.isnan(data), 0, data)


def draw_rectangles(img, rectangles, colors, thickness=2):
    """在图像上绘制矩形框"""
    for rect, color in zip(rectangles, colors):
        x, y, w, h = map(int, rect)
        cv.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def process_demo(sequence_name, config):
    """处理单个序列的demo绘制"""
    save_dir = os.path.join(config["save_dir"], sequence_name)
    if makedir(save_dir):
        filenames = read_filenames(config["image_path"])
        gts = load_data(config["gt_path"], delimiter=config.get("gt_delimiter", ','))
        baselines = [load_data(path, delimiter=delimiter)
                     for path, delimiter in config["baselines"]]

        # 校验数据长度一致
        assert len(gts) == len(filenames) == len(baselines[0]), "数据长度不一致"

        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        for i, filename in enumerate(filenames):
            img = cv.imread(filename)
            draw_rectangles(
                img, [gts[i]] + [baseline[i] for baseline in baselines], colors, config["thickness"]
            )
            cv.imwrite(os.path.join(save_dir, os.path.basename(filename)), img)


def main_demo(task_name, sequences, configs):
    """绘制多个序列的demo"""
    for seq_name in sequences:
        print(f"Processing: {seq_name}")
        process_demo(seq_name, configs[task_name])
        print("Done")


if __name__ == '__main__':
    CONFIGS = {
        "lasot": {
            "save_dir": "/mnt/second/wangyipei/SOI/reference/lasot_demo",
            "image_path": lambda seq: f"/mnt/first/hushiyu/SOT/LaSOT/data/{seq.split('-')[0]}/{seq}",
            "gt_path": lambda seq: f"/mnt/first/hushiyu/SOT/LaSOT/data/{seq.split('-')[0]}/{seq}/groundtruth.txt",
            "baselines": [
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/KeepTrack/lasotSOI/test/results/KeepTrack/{seq}.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/ToMP/lasotSOI/test/results/ToMP/{seq}.txt", '\t'),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/OSTrack/lasotSOI/test/results/OSTrack/{seq}.txt", ',')
            ],
            "thickness": 2
        },
        "videocube": {
            "save_dir": "/mnt/second/wangyipei/SOI/demo",
            "image_path": lambda seq: f"/mnt/first/hushiyu/SOT/VideoCube/data/val/{seq}/frame_{seq}",
            "gt_path": lambda seq: f"/mnt/first/hushiyu/SOT/VideoCube/data/val/{seq}/result_{seq}.txt",
            "baselines": [
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/transKT/videocubeSOI/val/results/transKT_restart/{seq}.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/KeepTrack/videocubeSOI/val/results/KeepTrack_restart/{seq}.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/ToMP/videocubeSOI/val/results/ToMP_restart/{seq}.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/OSTrack/videocubeSOI/val/results/OSTrack_restart/{seq}.txt", ',')
            ],
            "thickness": 5
        },
        "got10k": {
            "save_dir": "/mnt/second/wangyipei/SOI/demo",
            "image_path": lambda seq: f"/mnt/first/hushiyu/SOT/GOT-10k/data/val/{seq}",
            "gt_path": lambda seq: f"/mnt/first/hushiyu/SOT/GOT-10k/data/val/{seq}/groundtruth.txt",
            "baselines": [
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/transKT/got10kSOI/val/results/{seq}/{seq}_001.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/KeepTrack/got10kSOI/val/results/KeepTrack/{seq.split('_')[-1]}.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/ToMP/got10kSOI/val/results/ToMP/{seq}.txt", '\t'),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/OSTrack/got10kSOI/val/results/OSTrack/{seq}.txt", '\t')
            ],
            "thickness": 2
        },
        "vot": {
            "save_dir": "/mnt/second/wangyipei/SOI/demo",
            "image_path": lambda seq: f"/mnt/first/hushiyu/SOT/VOTLT2019/data/{seq}/color",
            "gt_path": lambda seq: f"/mnt/first/hushiyu/SOT/VOTLT2019/data/{seq}/groundtruth.txt",
            "baselines": [
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/TransKT/votSOI/test/VOTLT2019/TransKTtune3/{seq}.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/KeepTrack/votSOI/test/VOTLT2019/KeepTrack/{seq}.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/ToMP/votSOI/test/VOTLT2019/ToMP/{seq}.txt", ','),
                (lambda seq: f"/mnt/second/wangyipei/SOI/tracker_result/OSTrack/votSOI/test/VOTLT2019/OSTrack/{seq}.txt", ',')
            ],
            "thickness": 2
        }
    }


    # 设置任务和序列
    TASK = "lasot"
    SEQUENCES = [
        "sheep-9", "tiger-6", "turtle-8", "bird-2", "robot-1", "tiger-4",
        "hat-1", "hand-9", "zebra-10", "umbrella-9", "volleyball-19", "person-5"
    ]

    # 运行任务
    main_demo(TASK, SEQUENCES, CONFIGS)
