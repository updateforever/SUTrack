import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

# from lib.test.evaluation.multi_object_wrapper import MultiObjectWrapper
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import torch
from soi.screening_util import find_local_maxima_v1, mask_image_with_boxes, iou4list, mask_image_with_boxes_online, find_local_maxima_2d
import json

def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False, run_soi=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only, run_soi) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, run_soi=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.run_soi = run_soi
        self.vis_heatmap = False

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
            self.soi_dir = '{}/{}/{}'.format(env.soi_path, self.name, self.parameter_name)  # wyp
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)
            self.soi_dir = '{}/{}/{}_{:03d}'.format(env.soi_path, self.name, self.parameter_name, self.run_id)  # wyp

        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None
        # wyp
        self.threshold = 0.25
        if self.run_soi == 2:  # step 2 save results in soi dir
            self.results_dir = '{}/{}/{}'.format(env.soi_save_dir, self.name, self.parameter_name)
        if self.run_soi == 3:  # step 3 run soi online
            self.results_dir = '{}/{}/{}'.format(env.soi_online_save_dir, self.name, self.parameter_name)
            self.soi_vis_dir = '{}/{}/{}'.format(env.soi_vis_dir, self.name, self.parameter_name)
            self.soi_online = True
            # {"frame_id": frame_id, "status": status, "track_iou_gt": track_iou_gt, "gt_coord": gt_coord, "mask_boxes": mask_boxes}
            self.last_soi_info = {"frame_id": 1, "status": "right", "track_iou_gt": [], "gt_coord": None, "mask_boxes": []}  # TODO 保存soi信息，为后续文本生成做准备
        if self.run_soi == 4:  # step 4 run soi refer infer
            self.results_dir = '{}/{}/{}'.format(env.soi_infer_dir, self.name, self.parameter_name)
        if self.run_soi == 5:  # step 5 run soi refer infer with vlm
            self.results_dir = '{}/{}/{}'.format(env.soi_infer_with_vlm_dir, self.name, self.parameter_name)
            
        else:
            self.soi_online = False
        

    def create_tracker(self, params):

        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': [],
                  'segmentation': [],
                  }
        if self.run_soi == 3:  # TODO 可视化热力图结果  {"frame_id": frame_id, "status": status, "track_iou_gt": track_iou_gt, "gt_coord": gt_coord, "mask_boxes": mask_boxes}
            output['status'] = []
            output['track_iou_gt'] = []  # 预测结果的iou
            output['gt_coord'] = []  # gt值？
            output['mask_boxes'] = []
            # output['search_area_box'] = []
            # output['target_candidate_scores'] = []
            # output['target_candidate_coords'] = []
            # output['tg_num'] = []            
            # output['candidate_boxes'] = []
            if 0:
                output['search_img'] = []
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])
        init_info['seq_name'] = seq.name
        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        if self.run_soi == 2: 
            mask_info = []  # 初始化 mask 信息存储列表
            mask_jsonl_path = os.path.join(self.soi_dir, f"{seq.name}_mask_info.jsonl")
            with open(mask_jsonl_path, "r") as f: 
                for line in f:  # 逐行读取 JSONL 文件
                    try:
                        mask_info.append(json.loads(line.strip()))  # 解析 JSON 并添加到列表
                    except json.JSONDecodeError:
                        print(f" Invalid JSON line in {mask_jsonl_path}, skipping...")
                assert len(mask_info) == len(seq.frames) - 1, f"Mask info length {len(mask_info)} does not match frame length {len(seq.frames) - 1} for {seq.name}."
            print(f" Loaded {len(mask_info)} mask records for {seq.name}.")

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)  # array
            # if frame_num == 460:  
            #     return output
            if self.run_soi == 2:  # step 2 run online with masked image
                for record in mask_info:
                    if record["frame_id"] == frame_num:
                        mask_boxes = record.get("mask_boxes", [])
                        gt_coord = record.get("gt_coord", None)
                        break
                debug_save_path = f"{self.soi_dir}/mask_vis/{seq.name}/{frame_num:06d}.jpg" if tracker.debug else None
                if frame_num == 1 and tracker.debug:
                    print(f"Debug save path: {debug_save_path}")
                if isinstance(mask_boxes, list) and mask_boxes:
                    image = mask_image_with_boxes(image, mask_boxes, gt_coord, debug_save_path=debug_save_path)
            elif self.run_soi == 3:
                debug_save_path = f"{self.soi_vis_dir}/{seq.name}/{frame_num:06d}.jpg"  # if tracker.debug else None
                image = mask_image_with_boxes_online(image, self.last_soi_info['mask_boxes'], self.last_soi_info['gt_coord'], self.last_soi_info['status'], debug_save_path=debug_save_path)
            
            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            # wyp
            if self.run_soi == 1:  # step 1 just run for soi frames offline
                frame_candidate_data = self.extract_candidate_data(
                    tracker.distractor_dataset_data, 
                    filte_mode="alpha",
                    th=self.threshold
                    )
                # _store_outputs(frame_candidate_data, {})
                self.process_masked_images(seq.name, 
                                      image, seq.ground_truth_rect[frame_num, :],
                                      frame_candidate_data, 
                                      out['target_bbox'],  
                                      frame_num,
                                      )
                if self.vis_heatmap:
                    self.visualize_heatmap(seq.name, frame_candidate_data, frame_num)
            elif self.run_soi == 3:
                frame_candidate_data = self.extract_candidate_data(
                    tracker.distractor_dataset_data, 
                    filte_mode="debug",
                    th=self.threshold
                    )
                self.last_soi_info = process_candidate_boxes_and_gt(seq.name, 
                                                                    None, frame_num, seq.ground_truth_rect[frame_num, :], 
                                                                    frame_candidate_data["candidate_boxes"], 
                                                                    iou_threshold=0.5, bound=(image.shape[1], image.shape[0]),
                                                                    soi_step=self.run_soi
                                                                    )
                _store_outputs(self.last_soi_info)  # {"frame_id": frame_id, "status": status, "track_iou_gt": track_iou_gt, "gt_coord": gt_coord, "mask_boxes": mask_boxes}
                if self.vis_heatmap:
                    self.visualize_heatmap(seq.name, frame_candidate_data, frame_num)

        for key in ['target_bbox', 'all_boxes', 'all_scores', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")

    def extract_candidate_data(self, data, filte_mode='alpha', th=0.25):
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

        # target_candidate_coords, target_candidate_scores, candidate_boxes = (
        #     find_local_maxima_v1(
        #         score_map.squeeze(), all_boxes, th=th, ks=5, threshold_type=filte_mode
        #     )
        # )
        target_candidate_coords, target_candidate_scores, candidate_boxes = find_local_maxima_2d(score_map.squeeze(), all_boxes)
        
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
                score_map=score_map,
            )
        return dict(
            search_area_box=search_area_box,
            target_candidate_scores=target_candidate_scores,
            target_candidate_coords=target_candidate_coords,
            tg_num=tg_num,
            candidate_boxes=candidate_boxes,
            score_map=score_map,
        )

    def process_masked_images(self, 
                              seq_name, image, gt_box, 
                              frame_candidate_data, track_result, 
                              frame_id, save_masked_img=False  # mask_path, 
                              ):
        """ 处理 Masked 图像，并确保 `masked_info.jsonl` 追加写入不会重复 """ 
        mask_path = self.soi_dir
        if save_masked_img:
            masked_save_path = f"{mask_path}/masked_img/{seq_name}/{frame_id:08d}.jpg"
            os.makedirs(os.path.dirname(masked_save_path), exist_ok=True)
            mask_image_with_boxes(image, gt_box, frame_candidate_data["candidate_boxes"], track_result, 
                                iou_threshold=0.7, fill_color=(0, 0, 0), need_save=True, save_path=masked_save_path)
        else:
            masked_save_path = os.path.join(mask_path, f"{seq_name}_mask_info.jsonl")
            existing_frames = set()
            # **检查 `masked_info.jsonl` 是否已有该帧数据**
            if os.path.exists(masked_save_path):
                with open(masked_save_path, "r") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            existing_frames.add(record.get("frame_id"))
                        except json.JSONDecodeError:
                            continue  # 忽略 JSON 解析错误的行
            
            # **如果 `frame_id` 已存在，跳过写入**
            if frame_id in existing_frames:
                print(f"⚠️ Frame {frame_id} already exists in {masked_save_path}, skipping write.")
                return

            # **否则，追加写入**
            process_candidate_boxes_and_gt(
                seq_name, masked_save_path, frame_id, gt_box, 
                frame_candidate_data["candidate_boxes"], 
                iou_threshold=0.5, bound=(image.shape[1], image.shape[0])
            )

    # def visualize_heatmap(self, seq_name, frame_candidate_data, frame_id):
    #     frame_search_img = (frame_candidate_data["search_img"].squeeze().cpu().numpy())
    #     # 归一化热力图并转换为颜色映射
    #     heatmap = frame_candidate_data["score_map"].squeeze().cpu().numpy()  # self.tracker.distractor_dataset_data["score_map"]
    #     heatmap = cv.normalize(heatmap, None, 0, 255, cv.NORM_MINMAX)
    #     heatmap = np.uint8(heatmap)
    #     heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    #     # 如果尺寸不一致，调整尺寸
    #     if (heatmap.shape[:2] != frame_search_img.shape[:2]):  
    #         heatmap = cv.resize(heatmap, (frame_search_img.shape[1], frame_search_img.shape[0]))
    #     # 融合热力图与原图
    #     heat_map_search_img = cv.addWeighted(heatmap, 0.5, frame_search_img, 0.5, 0)
    #     # 添加文本
    #     fontScale = (frame_search_img.shape[1] / 500 * 1.0)  # 通过图像高度动态设置字体大小
    #     target_candidate_scores = frame_candidate_data["target_candidate_scores"]
    #     font_face = cv.FONT_HERSHEY_SIMPLEX
    #     cv.putText(heat_map_search_img, "No.%06d" % (frame_id), (10, 20), font_face, 0.8, (0, 255, 0), 2)
    #     # 将 target_candidate_scores 转换为可显示的字符串
    #     if target_candidate_scores.numel() == 1:  # 只有一个元素
    #         score_text = "tc_scores: %.2f" % target_candidate_scores.item()
    #     else:  # 多个元素，转换为逗号分隔的字符串
    #         score_list = (target_candidate_scores.flatten().tolist())  # 转换为 Python 列表
    #         score_text = "tc_scores: " + ", ".join(["%.2f" % s for s in score_list])
    #     # 在图像上显示分数文本
    #     cv.putText(heat_map_search_img, score_text, (10, 50), font_face, 0.8, (0, 255, 0), 2)
    #     # 保存图像
    #     ca_save_path = f"{self.soi_dir}/heatmap_vis//{seq_name}/{frame_id:06d}.jpg"
    #     os.makedirs(os.path.dirname(ca_save_path), exist_ok=True)
    #     cv.imwrite(ca_save_path, heatmap)


    def visualize_heatmap(self, seq_name, frame_candidate_data, frame_id):
        frame_search_img = (frame_candidate_data["search_img"].squeeze().cpu().numpy())  # torch.Size([3, 378, 378])
        # 转为384 384 3 
        if frame_search_img.shape[0] == 3:
            frame_search_img = np.transpose(frame_search_img, (1, 2, 0))  # C,H,W -> H,W,C
            frame_search_img = (frame_search_img - frame_search_img.min()) / (frame_search_img.max() - frame_search_img.min() + 1e-8)
            frame_search_img = (frame_search_img * 255).astype(np.uint8)
        # 转换颜色通道为 BGR（OpenCV 默认格式）
        frame_search_img = cv.cvtColor(frame_search_img, cv.COLOR_RGB2BGR)
        
        # 归一化热力图并转换为颜色映射
        score_map = frame_candidate_data["score_map"].squeeze().cpu().numpy()  # self.tracker.distractor_dataset_data["score_map"]

        # 2. resize 到原图大小（保持数值，不是颜色）
        if score_map.shape != frame_search_img.shape[:2]:
            score_map = cv.resize(score_map, (frame_search_img.shape[1], frame_search_img.shape[0]), interpolation=cv.INTER_LINEAR)

        # 4. 归一化到 0~255
        score_map_norm = cv.normalize(score_map, None, 0, 255, cv.NORM_MINMAX)
        # 5. 转为 uint8 再着色
        score_map_uint8 = np.uint8(score_map_norm)
        heatmap = cv.applyColorMap(score_map_uint8, cv.COLORMAP_JET)

        if frame_search_img.dtype != np.uint8:
            frame_search_img = frame_search_img.astype(np.uint8)
        heat_map_search_img = cv.addWeighted(heatmap, 0.7, frame_search_img, 0.3, 0)
        # 添加文本
        fontScale = (frame_search_img.shape[1] / 500 * 1.0)  # 通过图像高度动态设置字体大小
        target_candidate_scores = frame_candidate_data["target_candidate_scores"]
        font_face = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(heat_map_search_img, "No.%06d" % (frame_id), (10, 20), font_face, 0.8, (0, 255, 0), 2)
        # 将 target_candidate_scores 转换为可显示的字符串
        if target_candidate_scores.numel() == 1:  # 只有一个元素
            score_text = "tc_scores: %.2f" % target_candidate_scores.item()
        else:  # 多个元素，转换为逗号分隔的字符串
            score_list = (target_candidate_scores.flatten().tolist())  # 转换为 Python 列表
            score_text = "tc_scores: " + ", ".join(["%.2f" % s for s in score_list])
        # 在图像上显示分数文本
        cv.putText(heat_map_search_img, score_text, (10, 50), font_face, 0.8, (0, 255, 0), 2)
        # 保存图像
        ca_save_path = f"{self.soi_dir}/heatmap_vis//{seq_name}/{frame_id:06d}.jpg"
        os.makedirs(os.path.dirname(ca_save_path), exist_ok=True)
        cv.imwrite(ca_save_path, heat_map_search_img)

def process_candidate_boxes_and_gt(seq_name, masked_save_path, frame_id, gt, candidate_boxes, iou_threshold=0.6, bound=None, soi_step=1):
    """ 处理候选框和 GT 框，确定哪些需要 Mask，按 JSONL 格式存储，每行代表一帧。 """
    track_box = [int(s) for s in candidate_boxes[0].squeeze(0)]  # 获取预测框（第一个候选框）
    other_box_tensors = candidate_boxes[1:]  # 其余候选框
    gt = [int(s) for s in gt]  # 处理 GT（真实目标框），转换格式

    # 计算预测框与 GT 的 IoU（交并比）
    track_iou_gt = iou4list(track_box, gt)
    
    # 计算其他候选框与 GT 的 IoU
    other_iou_gts = [iou4list([int(s) for s in box.squeeze(0)], gt) for box in other_box_tensors]
    
    # 判断是否存在 IoU 高于阈值的其它候选框
    exist_other_good_box = any(val >= iou_threshold for val in other_iou_gts)
    
    # 初始化需要 Mask 的框
    mask_boxes = []

    # ------------------------------ 
    # 根据不同情况决定 Mask 逻辑 
    # ------------------------------
    if track_iou_gt >= iou_threshold:  # 预测框 IoU 高，可能是正确或妥协
        status = "Compromise" if exist_other_good_box else "Correct"
    else:  # 预测框 IoU 低，可能漂移或失败
        mask_boxes.append({"x1": track_box[0], "y1": track_box[1], "x2": track_box[0] + track_box[2], "y2": track_box[1] + track_box[3]})
        if exist_other_good_box:  # 存在正确候选框，跟踪漂移
            status = "Drift"
            for box_tensor in other_box_tensors:  # 遮挡所有候选框
                ob = [int(s) for s in box_tensor.squeeze(0)]
                mask_boxes.append({"x1": ob[0], "y1": ob[1], "x2": ob[0] + ob[2], "y2": ob[1] + ob[3]})
        else:  # 无正确候选框，跟踪失败
            status = "Fail"
    if gt == [0, 0, 0, 0]:  # TODO 好像不是很匹配，0.0.0.0.的情况
        status = "absent"
    # 记录 GT 框信息  mask的时候去修复gt框的坐标
    gt_coord = {"x1": gt[0], "y1": gt[1], "x2": gt[0] + gt[2], "y2": gt[1] + gt[3]}
    # 组织 JSON 数据
    frame_mask_info = {"frame_id": frame_id, "status": status, "track_iou_gt": track_iou_gt, "gt_coord": gt_coord, "mask_boxes": mask_boxes}
    if soi_step == 1:
        # 以 JSONL 方式保存到文件 
        os.makedirs(os.path.dirname(masked_save_path), exist_ok=True)  # 确保目录存在
        with open(masked_save_path, "a") as f: f.write(json.dumps(frame_mask_info) + "\n")  # 追加写入 JSONL 文件
        return
    elif soi_step == 3:
        # 返回 frame_mask_info
        return frame_mask_info