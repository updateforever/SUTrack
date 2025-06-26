# SOI Pipeline for PyTracking

SOI Pipeline是集成在PyTracking框架中的单目标跟踪语义描述生成模块，用于在跟踪失败场景下生成目标的语义描述。

## 目录结构

```
soi_pipeline/
├── __init__.py              # 模块初始化
├── configs/
│   └── config.py           # 配置类
├── core/
│   ├── box_utils.py        # 边界框工具
│   ├── data_processor.py   # 数据处理器
│   └── frame_extractor.py  # 帧提取器
├── models/
│   ├── vlm_interface.py    # VLM推理接口
│   └── text_cleaner.py     # 文本清理器
├── pipelines/
│   ├── step3_pipeline.py   # Step 3 流程
│   ├── step4_pipeline.py   # Step 4 流程
│   └── step5_pipeline.py   # Step 5 流程
├── utils/
│   ├── analysis.py         # 结果分析
│   └── visualization.py    # 可视化工具
├── scripts/
│   └── analyze_results.py  # 分析脚本
├── main.py                 # 主入口
└── README.md              # 本文件
```

## 快速开始

### 1. 基本使用

```bash
# 在PyTracking根目录下运行
cd /path/to/pytracking

# 运行完整流程
python -m soi_pipeline.main \
  --dataset lasot \
  --soi-tracker-dir /path/to/tracker/soi/results \
  --detection-dir /path/to/detection/results \
  --status-dir /path/to/tracker/status \
  --api-key your-openai-api-key

# 运行特定步骤
python -m soi_pipeline.main --steps 3 \
  --soi-tracker-dir /path/to/tracker/results

python -m soi_pipeline.main --steps 4 5 \
  --api-key your-api-key
```

### 2. 使用本地模型

```bash
python -m soi_pipeline.main \
  --use-local-model \
  --model-path /path/to/qwen2.5-vl-model \
  --soi-tracker-dir /path/to/tracker/results
```

### 3. 自定义参数

```bash
python -m soi_pipeline.main \
  --dataset lasot \
  --output-dir ./my_soi_outputs \
  --iou-thresh-gt 0.6 \
  --frame-gap-threshold 15 \
  --min-vote-ratio 0.6 \
  --api-key your-key \
  --verbose
```

## 处理流程

### Step 3: 框过滤和SOI帧提取
- **Step 3.1**: 合并多个跟踪器和检测器的边界框，过滤重叠和低质量框
- **Step 3.2**: 基于跟踪器状态投票和场景变化分析提取关键SOI帧

### Step 4: VLM语义描述生成
- 使用视觉语言模型（本地Qwen2.5-VL或OpenAI API）生成结构化语义描述
- 生成4层级描述：位置特征、外观描述、动态状态、干扰物区分
- 支持模板图像引导的描述生成

### Step 5: 文本清理和后处理
- 使用GPT-4清理VLM输出，移除推测性语言
- 确保描述格式一致性和实用性
- 提高描述质量

## 配置说明

### 主要参数

```python
@dataclass
class Config:
    # 数据集
    dataset_name: str = 'lasot'
    
    # 输入路径
    soi_tracker_dir: str = ''      # SOI跟踪器结果目录
    detection_dir: str = ''        # 检测结果目录 
    status_dir: str = ''           # 跟踪器状态目录
    output_dir: str = './soi_outputs'  # 输出目录
    
    # 处理参数
    iou_thresh_gt: float = 0.5     # GT过滤IoU阈值
    iou_thresh_nms: float = 0.4    # NMS IoU阈值
    frame_gap_threshold: int = 30   # SOI帧间隔阈值
    min_vote_ratio: float = 0.5    # 最小投票比例
    
    # VLM配置
    use_local_model: bool = False  # 使用本地模型
    model_path: str = ''           # 本地模型路径
    api_key: str = ''              # API密钥
    use_template: bool = True      # 使用模板图像
    
    # 其他
    verbose: bool = True           # 详细输出
    save_visualizations: bool = False  # 保存可视化
```

## 输出格式

```
soi_outputs/
├── step3_1_filtered_boxes/          # 过滤后的边界框
│   └── sequence_name.jsonl         # 每行包含一帧的所有框
├── step3_2_soi_frames/              # SOI帧索引
│   └── sequence_name_soi_frames.jsonl  # SOI帧索引列表
├── step4_vlm_descriptions/          # VLM生成的描述
│   └── sequence_name_descriptions.jsonl  # 每行一个SOI帧的描述
├── step5_cleaned_descriptions/      # 清理后的描述
│   └── sequence_name_cleaned.jsonl     # 清理后的描述
├── soi_frame_statistics.csv        # SOI帧统计
└── description_quality_metrics.json # 描述质量指标
```

## 编程接口

### 直接使用核心组件

```python
from soi_pipeline import Config, DataProcessor, VLMEngine
from lib.test.evaluation import get_dataset

# 创建配置
config = Config(
    dataset_name='lasot',
    soi_tracker_dir='/path/to/soi/results',
    api_key='your-key'
)

# 加载数据集
dataset = get_dataset('lasot')

# 使用数据处理器
processor = DataProcessor(config)
for seq in dataset:
    filtered_boxes = processor.process_sequence(seq.name, seq.ground_truth_rect)

# 使用VLM引擎
vlm_engine = VLMEngine(config)
description = vlm_engine.generate(
    image_paths=['image.jpg'],
    prompt='Describe the target object...'
)
```

### 运行单个流程

```python
from soi_pipeline.pipelines.step3_pipeline import run_step3_box_filtering
from soi_pipeline.pipelines.step4_pipeline import run_step4_description_generation

# 运行Step 3
step3_output = run_step3_box_filtering(config, dataset)

# 运行Step 4
step4_output = run_step4_description_generation(config, dataset, step3_1_dir, step3_2_dir)
```

## 结果分析

```bash
# 生成分析报告
python -m soi_pipeline.main --analyze-only --output-dir ./soi_outputs

# 或使用独立脚本
python soi_pipeline/scripts/analyze_results.py --output-dir ./soi_outputs
```

生成的分析包括：
- SOI帧数量分布统计
- 不同序列的SOI帧提取情况
- VLM描述生成成功率
- 描述质量指标

## 依赖环境

### 基础依赖
```bash
pip install numpy pandas opencv-python tqdm openai
```

### 本地模型支持（可选）
```bash
pip install torch torchvision modelscope transformers
```

### 分析工具（可选）
```bash
pip install matplotlib seaborn
```

## 扩展和自定义

### 添加新的VLM模型

```python
# 在models/vlm_interface.py中扩展
class CustomVLMEngine(VLMEngine):
    def _generate_local(self, image_paths, prompt):
        # 实现自定义模型推理逻辑
        pass
```

### 自定义帧提取策略

```python
# 在core/frame_extractor.py中修改
class CustomFrameExtractor(FrameExtractor):
    def extract_soi_candidates(self, trackers, seq_len, ground_truth_rects):
        # 实现自定义提取逻辑
        pass
```

### 自定义分析指标

```python
# 在utils/analysis.py中添加
def analyze_custom_metrics(output_dir):
    # 实现自定义分析逻辑
    pass
```

## 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 确保在PyTracking根目录运行
   cd /path/to/pytracking
   python -m soi_pipeline.main
   ```

2. **数据集加载失败**
   ```bash
   # 检查lib.test.evaluation模块是否可用
   python -c "from lib.test.evaluation import get_dataset; print('OK')"
   ```

3. **API调用失败**
   ```bash
   # 检查API密钥和网络连接
   python -c "from openai import OpenAI; client = OpenAI(api_key='your-key'); print('OK')"
   ```

4. **本地模型加载失败**
   ```bash
   # 检查模型路径和依赖
   pip install modelscope transformers torch
   ```

### 调试模式

```bash
# 启用详细输出
python -m soi_pipeline.main --verbose

# 只处理少量序列进行测试
# 修改main.py中的dataset切片：dataset = dataset[:5]
```

## 性能优化

1. **使用API模式**：避免本地模型的显存占用
2. **批量处理**：一次处理多个序列
3. **增量处理**：支持断点续传，避免重复处理
4. **并行处理**：可以并行运行不同序列的Step 4

## 贡献指南

1. 保持与PyTracking框架的兼容性
2. 遵循现有的代码风格和命名规范
3. 添加适当的错误处理和日志输出
4. 更新相关文档和示例

这个SOI Pipeline模块设计为PyTracking框架的有机组成部分，提供了完整的语义描述生成能力，同时保持了良好的可扩展性和易用性。