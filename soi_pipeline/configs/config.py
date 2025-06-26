# pytracking/soi_pipeline/configs/config.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Config:
    """SOI Pipeline配置类"""
    # 数据集配置
    dataset_name: str = 'lasot'
   
    # 输入路径
    soi_tracker_dir: str = ''
    detection_dir: str = ''
    status_dir: str = ''
   
    # 输出路径
    output_dir: str = './soi_outputs'
   
    # Step 3 参数
    iou_thresh_gt: float = 0.5
    iou_thresh_nms: float = 0.4
    frame_gap_threshold: int = 30
    min_vote_ratio: float = 0.5
    center_thresh: float = 20.0
    scale_thresh: float = 0.3
    small_area_thresh: float = 0.00005
    min_box_size: int = 16
   
    # VLM配置
    use_local_model: bool = False
    model_path: str = ''
    api_key: str = ''
    use_template: bool = True
    temperature: float = 0.6
    max_tokens: int = 256
   
    # Step 5 配置
    # 文本清理
    enable_text_cleaning: bool = True
    
    # 反向验证
    enable_verification: bool = True
    verification_ref_mode: str = 'first'  # none, first, prev
    verification_levels: List[str] = None  # 验证级别，如 ["12", "123", "1234"]
    verification_iou_threshold: float = 0.25
    save_verification_visualizations: bool = False
    
    # Step 6 配置 - 人机认知差异探索
    # 数据筛选
    vlm_failure_threshold: float = 0.3  # VLM失败的IoU阈值
    min_samples_per_sequence: int = 1   # 每个序列最少样本数
    max_samples_per_sequence: int = 20000  # 每个序列最多样本数
    
    # 人类实验配置
    enable_confidence_rating: bool = True  # 启用信心评级
    enable_difficulty_rating: bool = True  # 启用难度评级
    human_iou_threshold: float = 0.5      # 人类成功判定的IoU阈值
    
    # Gradio界面配置
    gradio_server_port: int = 7860        # Gradio服务器端口
    gradio_share_link: bool = False       # 是否创建公共分享链接
    auto_save_interval: int = 30          # 自动保存间隔（秒）
    max_time_per_sample: int = 300        # 每个样本最大标注时间（秒）
    
    # 分析配置
    enable_statistical_tests: bool = True  # 启用统计显著性检验
    generate_failure_analysis: bool = True # 生成失败案例分析
    save_analysis_plots: bool = True       # 保存分析图表
    
    # IRB实验配置
    experiment_title: str = "Human-Machine Cognitive Difference Study"
    estimated_duration_per_sample: float = 0.5  # 每个样本预估时间（分钟）
    target_participant_count: int = 25           # 目标参与者数量
   
    # 处理选项
    verbose: bool = True
    save_visualizations: bool = False
    # ✅ 新增字段：是否跳过已存在结果
    skip_existing_results: bool = True

    # 额外配置字典，用于存储其他配置项
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        # 初始化验证级别
        if self.verification_levels is None:
            self.verification_levels = ["12", "123", "1234"]
        
        # 确保路径末尾没有斜杠
        if self.soi_tracker_dir and self.soi_tracker_dir.endswith('/'):
            self.soi_tracker_dir = self.soi_tracker_dir[:-1]
        
        if self.detection_dir and self.detection_dir.endswith('/'):
            self.detection_dir = self.detection_dir[:-1]
        
        if self.status_dir and self.status_dir.endswith('/'):
            self.status_dir = self.status_dir[:-1]
        
        if self.output_dir and self.output_dir.endswith('/'):
            self.output_dir = self.output_dir[:-1]
        
        # 验证API密钥或本地模型路径
        if not self.use_local_model and not self.api_key:
            print("⚠️ Warning: No API key provided. VLM features will be limited.")
        
        if self.use_local_model and not self.model_path:
            print("⚠️ Warning: No local model path provided. VLM features will be limited.")

    def get_step_output_dir(self, step_name: str) -> str:
        """获取特定步骤的输出目录"""
        import os
        step_dir = os.path.join(self.output_dir, f"step{step_name}")
        os.makedirs(step_dir, exist_ok=True)
        return step_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if not k.startswith('_')})
    
    def save(self, filepath: str):
        """保存配置到JSON文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """从JSON文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class BatchConfig:
    """批次配置类 - 专门用于数据分批"""
    # 基本批次设置
    batch_size: int = 200                    # 每批样本数量
    min_batch_size: int = 50                 # 最小批次大小
    max_batch_size: int = 500                # 最大批次大小
    
    # 分层策略
    stratify_by: str = 'sequence_name'       # 分层依据: sequence_name, iou_range, failure_type, none
    stratify_weights: Dict[str, float] = field(default_factory=dict)  # 自定义分层权重
    
    # 样本处理
    shuffle_samples: bool = True             # 是否打乱样本
    random_seed: int = 42                    # 随机种子，确保可重复性
    balance_classes: bool = True             # 是否平衡类别分布
    
    # 重叠验证配置
    create_overlap: bool = False             # 是否创建重叠验证批次
    overlap_ratio: float = 0.1               # 重叠比例 (0.0-1.0)
    overlap_batch_size: int = 50             # 重叠批次大小
    overlap_selection: str = 'random'        # 重叠选择策略: random, representative, difficult
    
    # 质量控制
    exclude_edge_cases: bool = True          # 排除边缘案例（IoU接近阈值）
    edge_case_threshold: float = 0.05        # 边缘案例阈值范围
    min_samples_per_category: int = 5        # 每个类别最少样本数
    max_samples_per_category: int = 1000     # 每个类别最多样本数
    
    # 难度分布控制
    difficulty_distribution: Dict[str, float] = field(default_factory=lambda: {
        'easy': 0.3,      # 简单样本比例
        'medium': 0.4,    # 中等样本比例
        'hard': 0.3       # 困难样本比例
    })
    
    # IoU范围分层配置
    iou_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        'very_low': (0.0, 0.1),
        'low': (0.1, 0.2),
        'medium': (0.2, 0.3),
        'high': (0.3, 0.4),
        'very_high': (0.4, 1.0)
    })
    
    # 失败类型权重
    failure_type_weights: Dict[str, float] = field(default_factory=lambda: {
        'low_iou': 0.5,
        'semantic_degradation': 0.3,
        'all_levels_failed': 0.2
    })
    
    # 批次命名和标识
    batch_prefix: str = 'batch'              # 批次文件前缀
    use_timestamp: bool = False              # 是否在批次名中包含时间戳
    custom_batch_names: List[str] = field(default_factory=list)  # 自定义批次名称
    
    # 导出选项
    save_batch_stats: bool = True            # 保存批次统计信息
    save_distribution_plots: bool = True     # 保存分布图表
    export_format: str = 'jsonl'             # 导出格式: jsonl, json, csv
    
    def __post_init__(self):
        """初始化后验证"""
        # 验证批次大小
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.min_batch_size > self.batch_size:
            self.min_batch_size = max(1, self.batch_size // 2)
            print(f"⚠️ Adjusted min_batch_size to {self.min_batch_size}")
        
        if self.max_batch_size < self.batch_size:
            self.max_batch_size = self.batch_size * 2
            print(f"⚠️ Adjusted max_batch_size to {self.max_batch_size}")
        
        # 验证分层策略
        valid_strategies = ['sequence_name', 'iou_range', 'failure_type', 'none']
        if self.stratify_by not in valid_strategies:
            raise ValueError(f"Invalid stratify_by: {self.stratify_by}. Must be one of {valid_strategies}")
        
        # 验证重叠比例
        if not 0.0 <= self.overlap_ratio <= 1.0:
            raise ValueError("Overlap ratio must be between 0 and 1")
        
        # 验证难度分布
        if abs(sum(self.difficulty_distribution.values()) - 1.0) > 0.01:
            print("⚠️ Warning: Difficulty distribution does not sum to 1.0")
        
        # 设置随机种子
        import random
        import numpy as np
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def get_stratify_categories(self, samples: List[Dict]) -> Dict[str, List]:
        """根据分层策略对样本分类"""
        from collections import defaultdict
        
        categories = defaultdict(list)
        
        if self.stratify_by == 'sequence_name':
            for sample in samples:
                seq_name = sample.get('sequence_name', 'unknown')
                categories[seq_name].append(sample)
        
        elif self.stratify_by == 'iou_range':
            for sample in samples:
                iou = sample.get('iou_1234', 0)
                for range_name, (min_iou, max_iou) in self.iou_ranges.items():
                    if min_iou <= iou < max_iou:
                        categories[range_name].append(sample)
                        break
                else:
                    categories['other'].append(sample)
        
        elif self.stratify_by == 'failure_type':
            for sample in samples:
                reasons = sample.get('failure_reasons', [])
                primary_reason = reasons[0] if reasons else 'unknown'
                
                # 归类失败类型
                if primary_reason.startswith('low_iou'):
                    category = 'low_iou'
                elif primary_reason == 'semantic_degradation':
                    category = 'semantic_degradation'
                elif primary_reason == 'all_levels_failed':
                    category = 'all_levels_failed'
                else:
                    category = 'other'
                
                categories[category].append(sample)
        
        else:  # stratify_by == 'none'
            categories['all'] = samples
        
        return dict(categories)
    
    def validate_batch_distribution(self, batches: List[List[Dict]]) -> Dict[str, Any]:
        """验证批次分布质量"""
        import numpy as np
        
        stats = {
            'total_batches': len(batches),
            'batch_sizes': [len(batch) for batch in batches],
            'size_statistics': {},
            'distribution_quality': {},
            'recommendations': []
        }
        
        batch_sizes = stats['batch_sizes']
        if batch_sizes:
            stats['size_statistics'] = {
                'mean': np.mean(batch_sizes),
                'std': np.std(batch_sizes),
                'min': min(batch_sizes),
                'max': max(batch_sizes),
                'target': self.batch_size
            }
            
            # 评估大小分布质量
            size_cv = np.std(batch_sizes) / np.mean(batch_sizes) if np.mean(batch_sizes) > 0 else 0
            if size_cv > 0.2:
                stats['recommendations'].append(f"High batch size variation (CV={size_cv:.2f}). Consider adjusting min_batch_size.")
        
        # 评估类别分布
        if self.stratify_by != 'none':
            category_distributions = []
            for batch in batches:
                categories = self.get_stratify_categories(batch)
                category_counts = {k: len(v) for k, v in categories.items()}
                category_distributions.append(category_counts)
            
            stats['distribution_quality']['category_balance'] = category_distributions
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BatchConfig':
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if not k.startswith('_')})


@dataclass 
class HumanExperimentConfig:
    """人类实验专用配置"""
    vlm_failure_threshold: float = 0.3
    min_samples_per_sequence: int = 1
    max_samples_per_sequence: int = 2000
    enable_confidence_rating: bool = True
    enable_difficulty_rating: bool = True
    save_intermediate_results: bool = True
    
    # UI配置
    show_template_box: bool = True         # 在模板图中显示目标框
    enable_comments: bool = True           # 启用评论功能
    show_progress_bar: bool = True         # 显示进度条
    enable_skip_option: bool = True        # 允许跳过样本
    show_batch_info: bool = True           # 显示批次信息
    enable_hotkeys: bool = True            # 启用快捷键
    
    # 质量控制
    min_bbox_area: int = 100              # 最小边界框面积
    max_annotation_time: int = 600        # 最大标注时间（秒）
    require_confidence_rating: bool = True # 必须提供信心评级
    enable_quality_checks: bool = True     # 启用质量检查
    min_bbox_overlap_with_gt: float = 0.1  # 与GT的最小重叠度（质量检查）
    
    # 批次特定配置
    enable_batch_randomization: bool = True  # 批次内随机化
    allow_batch_switching: bool = False      # 允许批次间切换
    save_progress_per_batch: bool = True     # 按批次保存进度
    batch_completion_reward: bool = False    # 批次完成奖励机制
    
    # 标注员配置
    annotator_id_required: bool = False      # 是否需要标注员ID
    enable_annotator_feedback: bool = True   # 启用标注员反馈
    track_annotation_patterns: bool = True   # 跟踪标注模式
    
    # 语义层级配置
    semantic_levels: Dict[int, str] = field(default_factory=lambda: {
        1: "Visual Only (Template Image)",
        2: "+ Position/Location Info",
        3: "+ Appearance/Visual Features",
        4: "+ Motion/Behavior Dynamics",
        5: "+ Environmental Context & Distractors",
        6: "Cannot Determine Target"
    })
    
    # 评分配置
    confidence_levels: Dict[int, str] = field(default_factory=lambda: {
        1: "Very Low Confidence",
        2: "Low Confidence",
        3: "Medium Confidence",
        4: "High Confidence",
        5: "Very High Confidence"
    })
    
    difficulty_levels: Dict[int, str] = field(default_factory=lambda: {
        1: "Very Easy",
        2: "Easy",
        3: "Medium",
        4: "Difficult",
        5: "Very Difficult"
    })
    
    # 实验设计配置
    use_practice_samples: bool = True         # 使用练习样本
    practice_sample_count: int = 5            # 练习样本数量
    enable_breaks: bool = True                # 启用休息提醒
    break_interval: int = 50                  # 每N个样本提醒休息
    max_continuous_annotation_time: int = 3600  # 最大连续标注时间（秒）
    
    # 数据验证和一致性
    inter_annotator_reliability: bool = False   # 启用标注员间一致性检验
    consistency_check_ratio: float = 0.1        # 一致性检查样本比例
    gold_standard_samples: List[str] = field(default_factory=list)  # 金标准样本ID
    
    def __post_init__(self):
        """初始化后处理"""
        # 验证阈值范围
        if self.vlm_failure_threshold < 0 or self.vlm_failure_threshold > 1:
            raise ValueError("VLM failure threshold must be between 0 and 1")
        
        # 验证样本数量
        if self.min_samples_per_sequence > self.max_samples_per_sequence:
            raise ValueError("Minimum samples cannot be greater than maximum samples")
        
        # 验证一致性检查比例
        if not 0.0 <= self.consistency_check_ratio <= 1.0:
            raise ValueError("Consistency check ratio must be between 0 and 1")
        
        # 验证最小边界框重叠度
        if not 0.0 <= self.min_bbox_overlap_with_gt <= 1.0:
            raise ValueError("Minimum bbox overlap with GT must be between 0 and 1")
    
    def get_experiment_metadata(self) -> Dict[str, Any]:
        """获取实验元数据"""
        import datetime
        
        return {
            "experiment_config": self.to_dict(),
            "created_timestamp": datetime.datetime.now().isoformat(),
            "version": "2.0",
            "config_type": "human_experiment",
            "estimated_completion_time_hours": (
                self.max_samples_per_sequence * 0.5 / 60  # 假设每样本30秒
            ),
            "quality_control_enabled": self.enable_quality_checks,
            "batch_support": True
        }
    
    def create_practice_session_config(self) -> Dict[str, Any]:
        """创建练习阶段配置"""
        return {
            "practice_mode": True,
            "sample_count": self.practice_sample_count,
            "show_feedback": True,
            "allow_retries": True,
            "show_ground_truth": True,
            "time_limit_per_sample": None,  # 练习阶段不限时
            "explanation_enabled": True
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HumanExperimentConfig':
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if not k.startswith('_')})
    
    def save(self, filepath: str):
        """保存配置到JSON文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'HumanExperimentConfig':
        """从JSON文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)