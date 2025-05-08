import json
import numpy as np
import torch
import math
import cv2 as cv
import random
import torch.nn.functional as F
from torch import tensor
from .tensor import TensorList
import os
import cv2

from .metrics import iou


def iou4list(boxA, boxB):
    """
    计算 boxA 和 boxB 的 IoU
    boxA, boxB 格式: [x, y, w, h]
    """
    xA1, yA1, wA, hA = boxA
    xA2, yA2 = xA1 + wA, yA1 + hA

    xB1, yB1, wB, hB = boxB
    xB2, yB2 = xB1 + wB, yB1 + hB

    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = wA * hA
    areaB = wB * hB

    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def find_local_maxima_v1(
    scores, all_boxes, ks=5, s=3, threshold_type="alpha", alpha=0.6, th=0.1, min_s=0.0
):
    """
    根据不同的阈值方式在热图中找到局部极大值点。

    参数：
        scores (tensor): 热图数据，形状为 (batch_size, channels, height, width)。
        all_boxes (tensor): 所有候选框信息，形状为 (batch_size, num_points, 4)。
        ks (int): 局部邻域（卷积核大小），定义了两个极大值之间的最小距离。
        threshold_type (str): 阈值类型，'basic' 使用最小得分阈值 `th`，'alpha' 使用最大得分比例阈值 `alpha`。
        alpha (float): 最大得分比例阈值，仅在 `threshold_type='alpha'` 时有效。
        th (float): 最小得分阈值，仅在 `threshold_type='basic'` 时有效。
        min_s (float): 用于平移得分的最小值，避免负数对计算的影响。

    返回：
        tuple: 包含以下三个部分：
            - coords_batch (list or tensor): 每个批次的局部极大值点坐标。
            - intensities_batch (list or tensor): 每个批次局部极大值点的得分。
            - boxes_batch (list or tensor): 每个批次局部极大值点对应的候选框。
    """
    # 1. 平移得分，确保得分非负
    scores = scores - min_s
    max_score = torch.max(scores)  # 最大得分，用于 alpha 阈值计算
    ndims = scores.ndimension()  # 维度判断

    # 2. 确保输入是 4D (batch_size, channels, height, width)
    if ndims == 2:
        scores = scores.view(1, 1, scores.shape[0], scores.shape[1])

    # 3. 使用最大池化查找局部极大值
    scores_max = F.max_pool2d(scores, kernel_size=ks, stride=s, padding=ks // 2)

    # 恢复到原始尺寸
    upsampled_scores_max = F.interpolate(
        scores_max, size=scores.shape[-2:], mode="nearest"
    )

    # 4. 根据阈值类型筛选局部极大值
    if threshold_type == "basic":
        # 直接根据阈值筛选
        peak_mask = (scores == upsampled_scores_max) & (scores >= th)  
    elif threshold_type == "alpha":
        # 按最大得分比例筛选
        peak_mask = ((scores == upsampled_scores_max) & (scores >= th) & (scores > alpha * max_score))  
    elif threshold_type == "debug":
        peak_mask = (scores == upsampled_scores_max) & (scores > alpha * max_score)
        # peak_mask = ((scores == upsampled_scores_max) & (scores > alpha * max_score)) | (scores >= th)  # 为了可视化debug  或操作（odtrack为啥一上来就没有高分）
    else:
        raise ValueError(
            f"Invalid threshold_type: {threshold_type}. Choose either 'basic' or 'alpha'."
        )

    # 5. 提取局部极大值的坐标和得分
    coords = torch.nonzero(peak_mask, as_tuple=False)  # 返回局部极大值的坐标 torch.Size([n, 4])
    intensities = scores[peak_mask]  # 提取对应的得分

    # 将 y 和 x 转换为展平后的索引
    y_coords = coords[:, 2]  # 第3个维度表示y坐标
    x_coords = coords[:, 3]  # 第4个维度表示x坐标
    H, W = peak_mask.shape[2], peak_mask.shape[3]  # 特征图高宽 (24, 24)
    flatten_indices = y_coords * W + x_coords  # 计算展平后的索引

    # 6. 按得分从高到低排序
    idx_maxsort = torch.argsort(-intensities)
    coords = coords[idx_maxsort]  # 排序后的坐标  n, 4
    intensities = intensities[idx_maxsort]  # 排序后的得分

    boxes_batch = []

    # 7. 处理批量数据情况
    if ndims == 4:
        coords_batch, intensities_batch = [], []
        for i in range(scores.shape[0]):
            # 筛选属于当前批次的数据
            mask = coords[:, 0] == i
            selected_coords = coords[mask, 2:]  # 当前批次的坐标 (height, width)
            selected_intensities = intensities[mask]  # 当前批次的得分

            # 使用选中坐标从 all_boxes 中提取对应的候选框
            selected_boxes = all_boxes[i, :, flatten_indices]

            # 将结果添加到对应列表中
            coords_batch.append(selected_coords)
            intensities_batch.append(selected_intensities)
            boxes_batch.append(selected_boxes)
    else:
        coords_batch = coords[:, 2:]  # 提取坐标 (height, width)
        intensities_batch = intensities  # 提取得分
        for i in range(coords.size(0)):
            coo = coords[i]
            boxes_temp = all_boxes[:, :, coo[2], coo[3]]  # 提取对应的候选框 1,4,n
            boxes_batch.append(boxes_temp)
    # 8. 返回局部极大值的坐标、得分和对应的候选框
    return coords_batch, intensities_batch, boxes_batch


def find_local_maxima_2d(
    score_map, all_boxes, ks=5, s=5, alpha=0.6, min_s=0.0
):
    """
    在 2D 热图中查找局部极大值点，并返回对应候选框（适配 all_boxes shape 为 (1, 4, H, W)）。

    参数：
        score_map (Tensor): (H, W) 热图分数图
        all_boxes (Tensor): (1, 4, H, W)，每个像素对应一个 box
        ks (int): max pooling kernel size
        s (int): max pooling stride（控制极大值最小距离）
        alpha (float): 相对最大分数的阈值（过滤弱峰）
        min_s (float): 分数下移常数，防止负值干扰

    返回：
        coords (Tensor): 极大值坐标 (N, 2)，格式 [y, x]
        scores (Tensor): 极大值得分 (N,)
        boxes (Tensor): 对应候选框 (N, 4)
    """
    H, W = score_map.shape
    score_map = score_map - min_s
    max_score = score_map.max().item()

    # 1. pooling 找稀疏极大值得分
    pooled = F.max_pool2d(score_map.view(1, 1, H, W), kernel_size=ks, stride=s, padding=ks // 2)
    pooled_vals = pooled.view(-1).unique()

    coords_list, scores_list, boxes_list = [], [], []

    # 2. 遍历 pooling 结果，回原图找坐标
    for val in pooled_vals:
        if val <= alpha * max_score:
            continue
        match_mask = (score_map == val)
        idx = torch.nonzero(match_mask, as_tuple=False)  # (N, 2)
        if idx.numel() == 0:
            continue

        scores_sel = score_map[match_mask]
        flat_inds = idx[:, 0] * W + idx[:, 1]  # 展平索引
        boxes_flat = all_boxes[0].reshape(4, -1)  # (4, H*W)
        box_sel = boxes_flat[:, flat_inds].T  # (N, 4)

        coords_list.append(idx)
        scores_list.append(scores_sel)
        boxes_list.append(box_sel)

    # 3. 拼接并排序（可选）
    if coords_list:
        coords = torch.cat(coords_list, dim=0)
        scores = torch.cat(scores_list, dim=0)
        boxes = torch.cat(boxes_list, dim=0)

        sort_idx = torch.argsort(-scores)
        return coords[sort_idx], scores[sort_idx], boxes[sort_idx]
    else:
        coords = torch.empty((0, 2), dtype=torch.long)
        scores = torch.empty((0,))
        boxes = torch.empty((0, 4))

    return coords, scores, boxes


def load_dump_seq_data_from_disk(path):
    """
    从磁盘加载序列数据。

    参数：
        path (str): 数据文件路径。

    返回：
        dict: 存储序列数据的字典。
    """
    data = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    return data


def dump_seq_data_to_disk(save_path, seq_name, seq_data):
    """
    将序列数据保存到磁盘。

    参数：
        save_path (str): 保存路径。
        seq_name (str): 序列名称。
        seq_data (dict): 序列数据。
    """
    data = load_dump_seq_data_from_disk(save_path)
    data[seq_name] = seq_data
    with open(save_path, "w") as f:
        json.dump(data, f)


def load_jsonl_file(path):
    """
    从 JSONL 文件中加载数据。

    参数：
        path (str): JSONL 文件路径。

    返回：
        list: 包含所有数据的列表。
    """
    data = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
    return data


def convert_to_serializable(obj):
    """
    将对象中的 Tensor 转换为 JSON 可序列化的类型。
    """
    if isinstance(obj, torch.Tensor):  # 如果是 Tensor，转换为列表或标量
        return obj.tolist() if obj.dim() > 0 else obj.item()
    elif isinstance(obj, dict):  # 如果是字典，递归转换每个值
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):  # 如果是列表，递归转换每个元素
        return [convert_to_serializable(v) for v in obj]
    else:  # 其他类型直接返回
        return obj


def append_to_jsonl_file(path, seq_name, seq_data):
    """
    向 JSONL 文件追加一条数据。

    参数：
        path (str): JSONL 文件路径。
        seq_name (str): 序列名称。
        seq_data (dict): 序列数据。
    """
    # 将 seq_data 转换为 JSON 可序列化的格式
    serializable_seq_data = convert_to_serializable(seq_data)

    # 组织记录并写入 JSONL 文件
    record = {"seq_name": seq_name, "seq_data": serializable_seq_data}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def determine_frame_state(tracking_data, tracker, seq, th=0.25):
    """
    判断当前帧的状态。

    参数：
        tracking_data (dict): 跟踪数据。
        tracker: 跟踪器对象。
        seq: 序列对象。
        th (float): 阈值，用于判断候选分数。

    返回：
        str: 帧的状态（如 'G', 'H', 'J', 'K'），如果无法判断返回 None。
    """
    visible = seq.target_visible[tracker.frame_num - 1]
    num_candidates = tracking_data["target_candidate_scores"].shape[0]
    state = None

    if num_candidates >= 2:
        max_candidate_score = tracking_data["target_candidate_scores"].max()

        # 计算候选框与目标框的距离
        anno_and_target_candidate_score_dists = torch.sqrt(
            torch.sum(
                (
                    tracking_data["target_anno_coord"]
                    - tracking_data["target_candidate_coords"]
                )
                ** 2,
                dim=1,
            ).float()
        )
        ids = torch.argsort(anno_and_target_candidate_score_dists)

        score_dist_pred_anno = anno_and_target_candidate_score_dists[ids[0]]
        sortindex_correct_candidate = ids[0]
        score_dist_anno_2nd_highest_score_candidate = (
            anno_and_target_candidate_score_dists[ids[1]]
            if num_candidates > 1
            else 10000
        )

        # 根据得分和距离判断状态
        if (
            num_candidates > 1
            and score_dist_pred_anno <= 2
            and score_dist_anno_2nd_highest_score_candidate > 4
            and sortindex_correct_candidate == 0
            and max_candidate_score < th
            and visible != 0
        ):
            state = "G"
        elif (
            num_candidates > 1
            and score_dist_pred_anno <= 2
            and score_dist_anno_2nd_highest_score_candidate > 4
            and sortindex_correct_candidate == 0
            and max_candidate_score >= th
            and visible != 0
        ):
            state = "H"
        elif (
            num_candidates > 1
            and score_dist_pred_anno > 4
            and max_candidate_score >= th
            and visible != 0
        ):
            state = "J"
        elif (
            num_candidates > 1
            and score_dist_pred_anno <= 2
            and score_dist_anno_2nd_highest_score_candidate > 4
            and sortindex_correct_candidate > 0
            and max_candidate_score >= th
            and visible != 0
        ):
            state = "K"

    return state


def determine_subseq_state(frame_state, frame_state_previous):
    """
    判断子序列的状态。

    参数：
        frame_state (str): 当前帧的状态。
        frame_state_previous (str): 上一帧的状态。

    返回：
        str: 子序列状态（如 'GH', 'HK'），如果无法判断返回 None。
    """
    if frame_state is not None and frame_state_previous is not None:
        return "{}{}".format(frame_state_previous, frame_state)
    else:
        return None


def extract_candidate_data(data, th=0.1):
    """
    提取候选数据。

    参数：
        data (dict): 跟踪数据字典。
        th (float): 阈值，用于选择候选分数。

    返回：
        tuple: 包含候选框数据的字典和候选数量。
    """
    search_area_box = data["search_area_box"]
    score_map = data["score_map"].cpu()
    all_boxes = data["all_scoremap_boxes"]

    target_candidate_coords, target_candidate_scores, candidate_boxes = (
        find_local_maxima_v1(
            score_map.squeeze(), all_boxes, th=th, ks=5, threshold_type="debug"
        )
    )

    tg_num = len(target_candidate_scores)

    if "x_dict" in data.keys():
        search_img = torch.tensor(data["x_dict"])  # data['x_dict'].tensors
        return dict(
            search_area_box=search_area_box,
            target_candidate_scores=target_candidate_scores,
            target_candidate_coords=target_candidate_coords,
            tg_num=tg_num,
            search_img=search_img,
            candidate_boxes=candidate_boxes,
        )
    return dict(
        search_area_box=search_area_box,
        target_candidate_scores=target_candidate_scores,
        target_candidate_coords=target_candidate_coords,
        tg_num=tg_num,
        candidate_boxes=candidate_boxes,
    )


def update_seq_data(seq_candidate_data, frame_candidate_data):
    """
    更新序列数据，将帧数据添加到序列数据中。

    参数：
        seq_candidate_data (dict): 序列候选数据，包含多个帧的候选信息。
        frame_candidate_data (dict): 当前帧的候选数据。
    """
    # 遍历每个键值对
    for key, val in frame_candidate_data.items():
        # 如果值是Tensor类型，将其转为float并转换为列表
        val = val.float().tolist() if torch.is_tensor(val) else val
        # 将当前帧的值添加到序列数据中
        seq_candidate_data[key].append(val)


def mask_image_with_boxes(
    image: np.ndarray,
    gt_box: list,
    candidate_boxes: list,
    track_result: np.ndarray,
    iou_threshold: float = 0.7,
    fill_color: tuple = (0, 0, 0),
    need_save: bool = False,
    save_path: str = None,
) -> np.ndarray:
    """
    对输入图像进行基于候选框的Mask操作，并可选地恢复GT区域。

    参数:
    ----
    image : np.ndarray
        原始图像 (H x W x 3)。
    gt_box : list
        Ground Truth框，格式 [x, y, w, h]。
    candidate_boxes : list
        候选框列表，每个候选框可为 [x, y, w, h]。
    track_result : np.ndarray
        跟踪器得到的预测结果，可用于和候选框比较IoU的判断。
    iou_func : callable
        计算 IoU 的函数，调用形式如 `iou_func(track_result, box_array, bound=...)`。
        其中 `track_result` 和 `box_array` 需要是 np.ndarray。
    iou_threshold : float
        IoU阈值，当候选框与 track_result 的IoU小于该值时，则进行Mask。
    fill_color : tuple
        遮挡颜色，例如(0, 0, 0)代表纯黑。
    need_save : bool
        是否需要把Mask后的图像写到磁盘。
    save_path : str
        如果 need_save=True，指定保存路径 (包含文件名)。

    返回:
    ----
    masked_frame : np.ndarray
        Mask操作完成后的图像 (H x W x 3)。
    """

    # 拷贝原图，避免修改原图像素
    masked_frame = image.copy()

    # 用于统计 Mask 计数（若需要的话）
    mask_count = 0

    # 遍历候选框，根据IoU判断是否遮挡
    for box_tensor in candidate_boxes:
        # box_tensor 可能是Torch张量，需要先转换为int类型的list或np数组
        temp_state = [int(s) for s in box_tensor.squeeze(0)]

        # 将候选框也转换为 np.ndarray 来匹配 iou_func 的输入要求
        box_array = np.array(temp_state, dtype=np.float32)[
            None, :
        ]  # shape变成 (1,4) 方便 iou_func
        # 计算IoU
        temp_iou_pred = iou(track_result, box_array)

        # 满足Mask条件则遮挡
        if temp_iou_pred < iou_threshold:
            x1, y1 = temp_state[0], temp_state[1]
            x2 = temp_state[0] + temp_state[2]
            y2 = temp_state[1] + temp_state[3]

            masked_frame[y1:y2, x1:x2] = fill_color
            mask_count += 1

    # 恢复GT区域
    gt_x1, gt_y1 = gt_box[0], gt_box[1]
    gt_x2, gt_y2 = gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
    masked_frame[gt_y1:gt_y2, gt_x1:gt_x2] = image[gt_y1:gt_y2, gt_x1:gt_x2]

    # 如果需要保存，则写盘
    if need_save and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, masked_frame)

    return masked_frame


def process_candidate_boxes_and_gt(
    seq_name,
    masked_save_path,
    frame_id,
    gt,
    candidate_boxes,
    bound,
    iou_threshold=0.5,
):
    """
    将 candidate_boxes(第一个是预测框, 后面是其它高置信度框),
    根据四种跟踪情况决定要Mask哪些框，并且在mask_boxes列表最后附上GT框。
    """
    # 1) 第一个是预测框
    track_box_tensor = candidate_boxes[0]
    track_box = [int(s) for s in track_box_tensor.squeeze(0)]  # [x,y,w,h]

    # 2) 其它候选框
    other_box_tensors = candidate_boxes[1:]

    # 3) 计算 track_iou_gt
    gt = [int(s) for s in gt]
    track_iou_gt = iou4list(track_box, gt)

    # 4) 计算其它候选框与 GT 的 IoU，以判断是否存在正确候选框
    other_iou_gts = []
    for box_tensor in other_box_tensors:
        ob = [int(s) for s in box_tensor.squeeze(0)]  # [x,y,w,h]
        other_iou_gts.append(iou4list(ob, gt))

    # 是否存在其它与GT IoU >= 阈值
    exist_other_good_box = any(val >= iou_threshold for val in other_iou_gts)

    mask_boxes = []  # 要被遮挡的区域(列表)

    # ------------------------------
    # 根据四种情况做逻辑分支
    # ------------------------------
    if track_iou_gt >= iou_threshold:
        # 情况 A：跟踪正确 或 妥协
        if exist_other_good_box:
            # 跟踪妥协
            status = "Compromise"
            # 策略1：保留预测框(不mask)，mask 其它候选
            # for i, box_tensor in enumerate(other_box_tensors):
            #     ob = [int(s) for s in box_tensor.squeeze(0)]
            #     x1, y1, x2, y2 = ob[0], ob[1], ob[0] + ob[2], ob[1] + ob[3]
            #     mask_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

            # 策略2：不mask，因为这时候没有跟踪失败？
            pass
        else:
            # 跟踪正确
            # 策略示例：不mask任何框(预测框和其它都保留)
            status = "Correct"

    else:
        # 情况 B：跟踪漂移 或 失败
        x1, y1 = track_box[0], track_box[1]
        x2, y2 = track_box[0] + track_box[2], track_box[1] + track_box[3]
        mask_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        if exist_other_good_box:
            # 跟踪漂移：预测是错的，同时存在gt候选框（跟踪器对gt保持高关注度，可能会修复回去）
            status = "Drift"
            # 策略示例：将预测框 + 所有其它候选都 mask
            for i, box_tensor in enumerate(other_box_tensors):
                ob = [int(s) for s in box_tensor.squeeze(0)]
                x1, y1, x2, y2 = ob[0], ob[1], ob[0] + ob[2], ob[1] + ob[3]
                mask_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        else:
            # 跟踪失败：预测是错的，但没有其它候选框，只mask预测框（跟踪器失去对gt的关注度，很难修复）
            status = "Fail"

    # ------------------------------
    # 最后将 GT 框也加入到 mask_boxes (通常是用来在推理时恢复 GT 区域)
    # ------------------------------
    gt_x1, gt_y1 = gt[0], gt[1]
    gt_x2, gt_y2 = gt_x1 + gt[2], gt_y1 + gt[3]

    # -------------- 记录 GT 坐标信息 --------------
    gt_coord = {"x1": gt_x1, "y1": gt_y1, "x2": gt_x2, "y2": gt_y2}

    # 将信息存储为一个 dict
    frame_mask_info = {
        # "seq_name": seq_name,
        "frame_id": frame_id,
        "status": status,  # 当前帧属于哪种情况
        "track_iou_gt": track_iou_gt,  # 预测框与GT的IoU
        "gt_coord": gt_coord,  # 记录GT框坐标
        "mask_boxes": mask_boxes,  # 需要被遮挡的框
    }

    # ------------------------------
    # 保存到 masked_save_path
    # ------------------------------
    # 确保文件夹存在
    os.makedirs(os.path.dirname(masked_save_path), exist_ok=True)

    # 如果文件存在，读取现有数据
    if os.path.exists(masked_save_path):
        with open(masked_save_path, "r") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # 在 existing_data 中查找是否已存在同一个 seq_name + frame_id 的记录
    found = False
    for idx, record in enumerate(existing_data):
        # 根据业务需要，这里匹配 seq_name + frame_id
        if record.get("frame_id") == frame_mask_info["frame_id"]:
            # 找到同一帧记录，进行覆盖更新
            existing_data[idx] = frame_mask_info
            found = True
            break

    # 如果没有找到同一帧，则追加
    if not found:
        existing_data.append(frame_mask_info)

    # 写回 JSON 文件
    with open(masked_save_path, "w") as f:
        json.dump(existing_data, f, indent=4)


def find_mask_info_for_frame(mask_data, seq_name, frame_id):
    """
    在 mask_data (list of dict) 中查找指定 seq_name, frame_id 的记录,
    返回 (mask_boxes, gt_coord)。
    如果找不到，则返回 (None, None)。
    """
    for record in mask_data:
        if record["frame_id"] == frame_id:
            mask_boxes = record.get("mask_boxes", [])
            gt_coord = record.get("gt_coord", None)
            return mask_boxes, gt_coord
    return None, None


def mask_image_with_boxes(
    image: np.ndarray,
    mask_boxes: list,
    gt_coord: dict = None,
    fill_color: tuple = (0, 0, 0),
    debug_save_path: str = None,
) -> np.ndarray:
    """
    对图像进行遮挡: 先将 mask_boxes 对应区域填充为 fill_color，
    然后再把 gt_coord 区域恢复为原图像素，以防 GT 被误遮挡。

    参数:
    ----
    image : np.ndarray
        原始图像 (H x W x 3)。
    mask_boxes : list of dict
        干扰框列表，每个框包含 {"x1", "y1", "x2", "y2"}。
    gt_coord : dict
        GT 框坐标，如 {"x1":..., "y1":..., "x2":..., "y2":...}。
        如果为 None，则表示不做 GT 恢复。
    fill_color : tuple
        遮挡颜色，如 (0,0,0)。
    debug_save_path : str
        如果不为空，则将原图和Mask后图拼接对比并保存到此路径 (适合在服务器上调试)。

    返回:
    ----
    masked_frame : np.ndarray
        先遮挡、后恢复 GT 的最终图像。
    """
    masked_frame = image.copy()

    # 1) 先把干扰框全部填充为 fill_color
    for box in mask_boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        # 做坐标越界检查
        x1_clamp = max(0, min(x1, masked_frame.shape[1]))
        x2_clamp = max(0, min(x2, masked_frame.shape[1]))
        y1_clamp = max(0, min(y1, masked_frame.shape[0]))
        y2_clamp = max(0, min(y2, masked_frame.shape[0]))

        masked_frame[y1_clamp:y2_clamp, x1_clamp:x2_clamp] = fill_color

    # 2) 再恢复 GT 区域（如果有 gt_coord）
    if gt_coord is not None:
        gx1, gy1 = gt_coord["x1"], gt_coord["y1"]
        gx2, gy2 = gt_coord["x2"], gt_coord["y2"]

        gx1_clamp = max(0, min(gx1, masked_frame.shape[1]))
        gx2_clamp = max(0, min(gx2, masked_frame.shape[1]))
        gy1_clamp = max(0, min(gy1, masked_frame.shape[0]))
        gy2_clamp = max(0, min(gy2, masked_frame.shape[0]))

        masked_frame[gy1_clamp:gy2_clamp, gx1_clamp:gx2_clamp] = image[
            gy1_clamp:gy2_clamp, gx1_clamp:gx2_clamp
        ]

    # 3) 如果给定了 debug_save_path，则保存对比图
    if debug_save_path is not None and len(mask_boxes) > 0:
        # 左边原图，右边Mask后图
        vis = np.hstack((image, masked_frame))
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, vis)
        # print(f"[DEBUG] Saved debug comparison to {debug_save_path}")

    return masked_frame



def mask_image_with_boxes_online(
    image: np.ndarray,
    mask_boxes: list,
    gt_coord: dict = None,
    status: str = "normal",
    fill_color: tuple = (0, 0, 0),
    debug_save_path: str = None,
) -> np.ndarray:
    """
    对图像进行遮挡: 先将 mask_boxes 对应区域填充为 fill_color，
    然后再把 gt_coord 区域恢复为原图像素，以防 GT 被误遮挡。

    参数:
    ----
    image : np.ndarray
        原始图像 (H x W x 3)。
    mask_boxes : list of dict
        干扰框列表，每个框包含 {"x1", "y1", "x2", "y2"}。
    gt_coord : dict
        GT 框坐标，如 {"x1":..., "y1":..., "x2":..., "y2":...}。
        如果为 None，则表示不做 GT 恢复。
    fill_color : tuple
        遮挡颜色，如 (0,0,0)。
    debug_save_path : str
        如果不为空，则将原图和Mask后图拼接对比并保存到此路径 (适合在服务器上调试)。

    返回:
    ----
    masked_frame : np.ndarray
        先遮挡、后恢复 GT 的最终图像。
    """
    if status == "absent":
        return image  # 如果是 absent，则直接返回原图

    masked_frame = image.copy()

    # 1) 先把干扰框全部填充为 fill_color
    for box in mask_boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        # 做坐标越界检查
        x1_clamp = max(0, min(x1, masked_frame.shape[1]))
        x2_clamp = max(0, min(x2, masked_frame.shape[1]))
        y1_clamp = max(0, min(y1, masked_frame.shape[0]))
        y2_clamp = max(0, min(y2, masked_frame.shape[0]))

        masked_frame[y1_clamp:y2_clamp, x1_clamp:x2_clamp] = fill_color

    # 2) 再恢复 GT 区域（如果有 gt_coord）
    if gt_coord is not None:
        gx1, gy1 = gt_coord["x1"], gt_coord["y1"]
        gx2, gy2 = gt_coord["x2"], gt_coord["y2"]

        gx1_clamp = max(0, min(gx1, masked_frame.shape[1]))
        gx2_clamp = max(0, min(gx2, masked_frame.shape[1]))
        gy1_clamp = max(0, min(gy1, masked_frame.shape[0]))
        gy2_clamp = max(0, min(gy2, masked_frame.shape[0]))

        masked_frame[gy1_clamp:gy2_clamp, gx1_clamp:gx2_clamp] = image[
            gy1_clamp:gy2_clamp, gx1_clamp:gx2_clamp
        ]

    # 3) 如果给定了 debug_save_path，则保存对比图
    if debug_save_path is not None and len(mask_boxes) > 0:
        # 左边原图，右边Mask后图
        vis = np.hstack((image, masked_frame))
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        # cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        # print(f"[DEBUG] Saved debug comparison to {debug_save_path}")

    return masked_frame


def file_exists(file_path):
    """检查指定文件是否存在"""
    return os.path.exists(file_path)

# ===============================
# 推理阶段示例调用
# ===============================

if __name__ == "__main__":
    seq_name = "robot-1"
    frame_id = 1
    image_path = "/path/to/frame1.jpg"
    mask_json_path = "/mnt/second/wangyipei/SOI/nips25/org_results/lasot/test/odtrack/results/robot-1/masked_info.json"

    masked_image = inference_with_mask(
        seq_name=seq_name,
        frame_id=frame_id,
        image_path=image_path,
        mask_json_path=mask_json_path,
    )

    if masked_image is not None:
        # 显示或保存 Mask 后的图像
        cv2.imshow("Masked Image", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 或者保存到文件
        masked_save_path = "/path/to/masked_frame1.jpg"
        cv2.imwrite(masked_save_path, masked_image)
