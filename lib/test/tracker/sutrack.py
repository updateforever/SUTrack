from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d
from lib.models.sutrack import build_sutrack
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box, clip_box_batch
import clip
import numpy as np
import math

# ============================================================================
# 通用文本标签集成代码块 - 直接复制到任何跟踪器中使用
# ============================================================================
import json
import os
from typing import Dict, Optional

class TextLabelIntegration:
    """
    简洁文本标签集成类 - 30帧保质期管理
    """
    
    def __init__(self, 
                 text_level_control: int = 1234,
                 text_effective_frames: int = 30,
                 text_base_path: str = "/home/wyp/project/SUTrack/soi_outputs/lasot_old/step4_vlm_descriptions"):
        """
        初始化文本标签功能
        
        Args:
            text_level_control: 文本级别控制 (12, 123, 1234)
            text_effective_frames: 文本保质期帧数
            text_base_path: 文本数据基础路径
        """
        self.text_level_control = text_level_control
        self.text_effective_frames = text_effective_frames
        self.text_base_path = text_base_path
        
        # 状态变量
        self.text_descriptions = {}
        self.current_text_frame = None
        self.current_text = None
        self.text_loaded = False
        
        print(f"Text integration initialized - Level: {text_level_control}, Frames: {text_effective_frames}")
    
    def load_text_data(self, sequence_name: str) -> bool:
        """加载序列文本数据"""
        if self.text_loaded:
            return True
            
        jsonl_file = os.path.join(self.text_base_path, f"{sequence_name}_descriptions.jsonl")
        
        if not os.path.exists(jsonl_file):
            print(f"Text file not found: {jsonl_file}")
            self.text_loaded = True
            return False
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    frame_idx = data.get('frame_idx')
                    vlm_output = data.get('vlm_output_cleaned') or data.get('vlm_output')
                    
                    if frame_idx is not None and vlm_output is not None:
                        self.text_descriptions[frame_idx] = vlm_output
            
            self.text_loaded = True
            print(f"Loaded {len(self.text_descriptions)} text descriptions for {sequence_name}")
            return True
            
        except Exception as e:
            print(f"Error loading text: {e}")
            self.text_loaded = True
            return False
    
    def get_text_for_frame(self, frame_id: int) -> Optional[str]:
        """获取当前帧的文本描述"""
        # 检查当前帧是否有新文本
        if frame_id in self.text_descriptions:
            self.current_text_frame = frame_id
            vlm_output = self.text_descriptions[frame_id]
            self.current_text = self._combine_text_levels(vlm_output)
            return self.current_text
        
        # 检查保质期
        if (self.current_text_frame is not None and 
            self.current_text is not None and
            frame_id <= self.current_text_frame + self.text_effective_frames):
            return self.current_text
        
        # 超过保质期，清除
        self.current_text_frame = None
        self.current_text = None
        return None
    
    def _combine_text_levels(self, vlm_output: Dict) -> str:
        """组合文本级别"""
        if not vlm_output:
            return ""
        
        text_parts = []
        level_str = str(self.text_level_control)
        
        if '1' in level_str and 'level1' in vlm_output:
            text_parts.append(vlm_output['level1'].strip())
        
        if '2' in level_str and 'level2' in vlm_output:
            text_parts.append(vlm_output['level2'].strip())
        
        if '3' in level_str and 'level3' in vlm_output:
            text_parts.append(vlm_output['level3'].strip())
        
        if '4' in level_str and 'level4' in vlm_output:
            level4 = vlm_output['level4']
            if isinstance(level4, list):
                text_parts.extend([t.strip() for t in level4 if t.strip()])
            elif isinstance(level4, str):
                text_parts.append(level4.strip())
        
        return ' '.join(text_parts)



class SUTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SUTRACK, self).__init__(params)
        network = build_sutrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        # mapping similar datasets
        if 'GOT10K' in DATASET_NAME:
            DATASET_NAME = 'GOT10K'
        if 'LASOT' in DATASET_NAME:
            DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME:
            DATASET_NAME = 'TNL2K'

        # multi modal vision
        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT
        print("MULTI_MODAL_VISION is: ", self.multi_modal_vision)

        #multi modal language
        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT
        print("MULTI_MODAL_LANGUAGE is: ", self.multi_modal_language)

        #using nlp information
        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT
        print("USE_NLP is: ", self.use_nlp)

        self.task_index_batch = None
        self.run_with_soi_refer = params.run_with_soi_refer
        if self.run_with_soi_refer:
            self.text_integration = TextLabelIntegration(
                text_level_control=1234,
                text_effective_frames=30
            )


    def initialize(self, image, info: dict):

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)
        z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and (template.size(1) == 3):
            template = torch.cat((template, template), axis=1)
        self.template_list = [template] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)]
        self.frame_id = 0

        # language information
        if self.multi_modal_language:
            if self.use_nlp:
                init_nlp = info.get("init_nlp")
            else:
                init_nlp = None
            text_data, _ = self.extract_token_from_nlp_clip(init_nlp)
            text_data = text_data.unsqueeze(0).to(template.device)
            with torch.no_grad():
                self.text_src = self.network.forward_textencoder(text_data=text_data)
        else:
            self.text_src = None
        
        # 加载文本数据
        if self.run_with_soi_refer:
            seq_name = info.get('seq_name') or getattr(self, 'seq_name', None)
            if seq_name:
                self.text_integration.load_text_data(seq_name)


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        # get search box wyp
        search_box = self.search_box(image, self.state, self.params.search_factor)  

        if self.multi_modal_vision and (search.size(1) == 3):
            search = torch.cat((search, search), axis=1)
        search_list = [search]

        if self.run_with_soi_refer:
            # 检查文本更新
            current_text = self.text_integration.get_text_for_frame(self.frame_id)
            
            if current_text:
                # 有文本标签，更新文本特征
                if current_text != getattr(self, '_last_text', ''):
                    # print(f"Frame {self.frame_id}: Using text - {current_text}")
                    text_data, _ = self.extract_token_from_nlp_clip(current_text)
                    text_data = text_data.unsqueeze(0).to(self.template_list[0].device)
                    with torch.no_grad():
                        self.soi_text_src = self.network.forward_textencoder(text_data=text_data)
                    self._last_text = current_text

                # run the encoder
                with torch.no_grad():
                    enc_opt = self.network.forward_encoder(self.template_list,
                                                    search_list,
                                                    self.template_anno_list,
                                                    self.soi_text_src,
                                                    self.task_index_batch)
            else:
                # 没有文本标签，使用默认特征
                if hasattr(self, '_last_text'):
                    # print(f"Frame {self.frame_id}: Using default text")
                    delattr(self, '_last_text')

                # run the encoder
                with torch.no_grad():
                    enc_opt = self.network.forward_encoder(self.template_list,
                                                    search_list,
                                                    self.template_anno_list,
                                                    self.text_src,
                                                    self.task_index_batch)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=enc_opt)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True: # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response,
                                                                   out_dict['offset_map'],
                                                                   return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        pre_state = self.state
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # wyp
        all_scoremap_boxes = self.network.decoder.cal_bbox_for_all_scores(response, out_dict['size_map'], out_dict['offset_map'])  # 1,4,576
        all_scoremap_boxes = all_scoremap_boxes.view(1, 4, 24, 24) * self.params.search_size / resize_factor  # 放缩 
        all_state = self.map_box_back_batch(all_scoremap_boxes, resize_factor, pre_state)  # 映射回原图的框
        all_state = clip_box_batch(all_state, H, W, margin=10)  # torch.Size([1, 4, 24, 24])
        self.distractor_dataset_data = dict(score_map=response,
                                    # sample_pos=sample_pos[scale_ind, :],
                                    sample_scale=resize_factor,
                                    search_area_box=search_box, 
                                    x_dict=x_patch_arr,
                                    all_scoremap_boxes=all_state
                                    )

        # update the template
        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                if self.multi_modal_vision and (template.size(1) == 3):
                    template = torch.cat((template, template), axis=1)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        # for debug
        # if image.shape[-1] == 6:
        #     image_show = image[:,:,:3]
        # else:
        #     image_show = image
        # if self.debug == 1:
        #     x1, y1, w, h = self.state
        #     image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
        #     cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
        #     cv2.imshow('vis', image_BGR)
        #     cv2.waitKey(1)

        return {"target_bbox": self.state,
                "best_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    # def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
    #     cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
    #     cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
    #     half_side = 0.5 * self.params.search_size / resize_factor
    #     cx_real = cx + (cx_prev - half_side)
    #     cy_real = cy + (cy_prev - half_side)
    #     return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def extract_token_from_nlp_clip(self, nlp):
        if nlp is None:
            nlp_ids = torch.zeros(77, dtype=torch.long)
            nlp_masks = torch.zeros(77, dtype=torch.long)
        else:
            nlp_ids = clip.tokenize(nlp).squeeze(0)
            nlp_masks = (nlp_ids == 0).long()
        return nlp_ids, nlp_masks

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float, pre_state):
        """
        输入 pred_box 可以为 (1, 4, 24, 24) 或 (1, 4, 576)
        """
        cx_prev, cy_prev = pre_state[0] + 0.5 * pre_state[2], pre_state[1] + 0.5 * pre_state[3]

        # 适配不同输入维度
        if pred_box.ndim == 4:  # 形状 (1, 4, 24, 24)
            cx, cy, w, h = pred_box[:, 0, :, :], pred_box[:, 1, :, :], pred_box[:, 2, :, :], pred_box[:, 3, :, :]
        elif pred_box.ndim == 3:  # 形状 (1, 4, 576)
            cx, cy, w, h = pred_box[:, 0, :], pred_box[:, 1, :], pred_box[:, 2, :], pred_box[:, 3, :]
            # 将 (1, 4, 576) 转换为 (1, 4, 24, 24) 形状
            batch_size, channels, _ = pred_box.shape
            cx, cy, w, h = [x.view(batch_size, 24, 24) for x in (cx, cy, w, h)]
        else:
            raise ValueError(f"Unsupported pred_box shape: {pred_box.shape}")

        # 计算半边长度
        half_side = 0.5 * self.params.search_size / resize_factor

        # 计算真实的中心坐标 (cx_real, cy_real)
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)

        # 重新组合框: (xmin, ymin, w, h)
        mapped_box = torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=1)  # 输出 shape: (1, 4, 24, 24)

        return mapped_box

    
    def search_box(self, im, state, search_area_factor):
        # 搜索框坐标
        if not isinstance(state, list):
            x, y, w, h = state.tolist()
        else:
            x, y, w, h = state
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)  # 模板区域sz
        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz
        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz
        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)
        im_crop = [x1+x1_pad, y1+y1_pad, x2 - x2_pad -x1 - x1_pad, y2 - y2_pad - y1 - y1_pad] # x,y,h,w

        return im_crop
    
def get_tracker_class():
    return SUTRACK


