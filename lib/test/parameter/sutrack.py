from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.sutrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/sutrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    params.yaml_name = yaml_name
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/sutrack/%s/SUTRACK_ep%04d.pth.tar" %
    #                                  (yaml_name, cfg.TEST.EPOCH))

    params.checkpoint = "/home/wyp/project/SUTrack/SUTRACK_ep0180.pth.tar"  # sutrack-l 384
    # params.checkpoint = "/home/wyp/project/SUTrack/SUTRACK_b384_ep0180.pth.tar"  # sutrack-b 384

    # whether to save boxes from all queries
    params.save_all_boxes = False

    # wyp
    # ============================================================================
    # Parameters for SOI Text and VLM Grounding Integration
    # ============================================================================
    
    # Enable/Disable SOI text description module
    # This module provides dynamic text descriptions for grounding
    params.run_with_soi_refer = False

    # Enable/Disable the VLM grounding correction module
    params.run_with_grounding = True

    # VLM API Configuration
    # API type: "generic", "gpt4v", "qwenvl"
    params.grounding_api_type = "qwenvl"  
    
    # API endpoint URL
    params.grounding_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # API key for commercial services (GPT-4V, Qwen-VL)
    params.grounding_api_key = ''  # Set your API key here if using GPT-4V or Qwen-VL
    
    # Qwen-VL specific parameters (following official examples)
    params.qwen_model_id = "qwen2.5-vl-72b-instruct"  # or "qwen-vl-plus" for faster inference
    params.qwen_min_pixels = 512 * 28 * 28  # Minimum image resolution
    params.qwen_max_pixels = 2048 * 28 * 28  # Maximum image resolution
    
    # Grounding trigger conditions
    params.grounding_confidence_threshold = 0.4  # Lower threshold for more frequent grounding
    params.soi_frame_interval = 30  # Fixed interval grounding
    
    # Box fusion parameters
    params.grounding_fusion_weight = 0.7  # Weight for grounding vs tracking
    params.grounding_min_iou = 0.2  # Minimum IoU for fusion vs replacement
    
    # Text expiration control for ablation studies
    params.grounding_text_expiration = True  # Enable text expiration  无效参数
    params.grounding_text_expiration_frames = 0  # 30-frame validity period
    
    # ============================================================================
    
    return params
