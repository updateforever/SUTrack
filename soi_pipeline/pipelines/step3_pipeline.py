# # pytracking/soi_pipeline/pipelines/step3_pipeline.py
# import os
# import json
# from tqdm import tqdm
# from ..core.data_processor import DataProcessor
# from ..core.frame_extractor import FrameExtractor


# def run_step3_box_filtering(config, dataset) -> str:
#     """运行Step 3.1: 边界框过滤和合并（按序列跳过）"""
#     print("🔄 Starting Step 3.1: Box filtering and merging")
    
#     output_dir = os.path.join(config.output_dir, "step3_1_filtered_boxes")
#     os.makedirs(output_dir, exist_ok=True)
    
#     processor = DataProcessor(config)
    
#     for seq in tqdm(dataset, desc="Processing sequences"):
#         seq_name = seq.name
#         output_path = os.path.join(output_dir, f"{seq_name}.jsonl")

#         if config.skip_existing_results and os.path.exists(output_path):
#             if config.verbose:
#                 print(f"⏩ Skipped {seq_name}: already exists.")
#             continue
        
#         try:
#             filtered_boxes = processor.process_sequence(seq_name, seq.dataset, seq.ground_truth_rect)
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 for frame_boxes in filtered_boxes:
#                     f.write(json.dumps(frame_boxes, ensure_ascii=False) + '\n')
#             if config.verbose:
#                 print(f"✅ {seq_name}: processed {len(filtered_boxes)} frames")
#         except Exception as e:
#             print(f"❌ Failed to process {seq_name}: {e}")
#             continue
    
#     print(f"✅ Step 3.1 completed. Results saved to {output_dir}")
#     return output_dir


# def run_step3_frame_extraction(config, dataset, step3_1_dir: str) -> str:
#     """运行Step 3.2: SOI帧提取（按序列跳过）"""
#     print("🔄 Starting Step 3.2: SOI frame extraction")
    
#     output_dir = os.path.join(config.output_dir, "step3_2_soi_frames")
#     os.makedirs(output_dir, exist_ok=True)
    
#     extractor = FrameExtractor(config)
    
#     for seq in tqdm(dataset, desc="Extracting SOI frames"):
#         seq_name = seq.name
#         step3_1_jsonl = os.path.join(step3_1_dir, f"{seq_name}.jsonl")
#         output_path = os.path.join(output_dir, f"{seq_name}_soi_frames.jsonl")

#         # 跳过已完成序列
#         if config.skip_existing_results and os.path.exists(output_path):
#             if config.verbose:
#                 print(f"⏩ Skipped {seq_name}: already exists.")
#             continue
        
#         if not os.path.exists(step3_1_jsonl):
#             if config.verbose:
#                 print(f"⚠️ Warning: No filtered boxes found for {seq_name}")
#             continue

#         try:
#             soi_frames = extractor.extract_soi_frames(
#                 seq_name=seq_name,
#                 dataset_name=seq.dataset,
#                 frames=seq.frames,
#                 ground_truth_rects=seq.ground_truth_rect,
#                 step3_1_jsonl=step3_1_jsonl
#             )
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 json.dump(soi_frames, f, ensure_ascii=False)
#             if config.verbose:
#                 print(f"✅ {seq_name}: extracted {len(soi_frames)} SOI frames")
#         except Exception as e:
#             print(f"❌ Failed to extract SOI frames for {seq_name}: {e}")
#             continue
    
#     print(f"✅ Step 3.2 completed. Results saved to {output_dir}")
#     return output_dir


# pytracking/soi_pipeline/pipelines/step3_pipeline.py
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from ..core.data_processor import DataProcessor
from ..core.frame_extractor import FrameExtractor


class Step3Statistics:
    """Step3统计和可视化类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计数据"""
        # Step 3.1 统计
        self.box_stats = {
            'soi_boxes_per_frame': [],
            'det_boxes_per_frame': [],
            'total_boxes_per_frame': [],
            'filtered_boxes_per_frame': [],
            'sequences': []
        }
        
        # Step 3.2 统计
        self.frame_stats = {
            'candidates_per_seq': [],
            'final_frames_per_seq': [],
            'filter_reasons': defaultdict(int),
            'sequences': []
        }
    
    def add_box_stats(self, seq_name: str, soi_counts: list, det_counts: list, 
                     total_counts: list, filtered_counts: list):
        """添加框统计数据"""
        self.box_stats['sequences'].append(seq_name)
        self.box_stats['soi_boxes_per_frame'].extend(soi_counts)
        self.box_stats['det_boxes_per_frame'].extend(det_counts)
        self.box_stats['total_boxes_per_frame'].extend(total_counts)
        self.box_stats['filtered_boxes_per_frame'].extend(filtered_counts)
    
    def add_frame_stats(self, seq_name: str, candidates_count: int, final_count: int, 
                       filter_reasons: dict):
        """添加帧统计数据"""
        self.frame_stats['sequences'].append(seq_name)
        self.frame_stats['candidates_per_seq'].append(candidates_count)
        self.frame_stats['final_frames_per_seq'].append(final_count)
        
        for reason, count in filter_reasons.items():
            self.frame_stats['filter_reasons'][reason] += count
    
    def plot_box_statistics(self, output_dir: str):
        """绘制框统计图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Step 3.1: Box Filtering Statistics', fontsize=16)
        
        # 1. SOI框数量分布
        axes[0, 0].hist(self.box_stats['soi_boxes_per_frame'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('SOI Boxes per Frame Distribution')
        axes[0, 0].set_xlabel('Number of SOI Boxes')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(self.box_stats['soi_boxes_per_frame']), 
                          color='red', linestyle='--', 
                          label=f'Mean: {np.mean(self.box_stats["soi_boxes_per_frame"]):.2f}')
        axes[0, 0].legend()
        
        # 2. 检测框数量分布
        axes[0, 1].hist(self.box_stats['det_boxes_per_frame'], bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Detection Boxes per Frame Distribution') 
        axes[0, 1].set_xlabel('Number of Detection Boxes')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(self.box_stats['det_boxes_per_frame']), 
                          color='red', linestyle='--',
                          label=f'Mean: {np.mean(self.box_stats["det_boxes_per_frame"]):.2f}')
        axes[0, 1].legend()
        
        # 3. 过滤前后对比
        x = range(len(self.box_stats['total_boxes_per_frame']))
        axes[1, 0].scatter(x[::100], [self.box_stats['total_boxes_per_frame'][i] for i in x[::100]], 
                          alpha=0.6, s=10, label='Before Filtering', color='orange')
        axes[1, 0].scatter(x[::100], [self.box_stats['filtered_boxes_per_frame'][i] for i in x[::100]], 
                          alpha=0.6, s=10, label='After Filtering', color='purple')
        axes[1, 0].set_title('Boxes Before vs After Filtering (Sampled)')
        axes[1, 0].set_xlabel('Frame Index (Sampled)')
        axes[1, 0].set_ylabel('Number of Boxes')
        axes[1, 0].legend()
        
        # 4. 过滤效果统计
        total_before = sum(self.box_stats['total_boxes_per_frame'])
        total_after = sum(self.box_stats['filtered_boxes_per_frame'])
        filter_ratio = (total_before - total_after) / total_before if total_before > 0 else 0
        
        categories = ['Original', 'Filtered']
        values = [total_before, total_after]
        colors = ['lightcoral', 'lightblue']
        
        bars = axes[1, 1].bar(categories, values, color=colors)
        axes[1, 1].set_title(f'Total Boxes: Filtering Effect\n(Filtered Ratio: {filter_ratio:.2%})')
        axes[1, 1].set_ylabel('Total Number of Boxes')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step3_1_box_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成统计报告
        self._generate_box_report(output_dir)
    
    def plot_frame_statistics(self, output_dir: str):
        """绘制帧统计图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Step 3.2: SOI Frame Extraction Statistics', fontsize=16)
        
        # 1. 候选帧数量分布
        axes[0, 0].hist(self.frame_stats['candidates_per_seq'], bins=30, alpha=0.7, color='orange')
        axes[0, 0].set_title('Candidate Frames per Sequence')
        axes[0, 0].set_xlabel('Number of Candidate Frames')
        axes[0, 0].set_ylabel('Number of Sequences')
        axes[0, 0].axvline(np.mean(self.frame_stats['candidates_per_seq']), 
                          color='red', linestyle='--',
                          label=f'Mean: {np.mean(self.frame_stats["candidates_per_seq"]):.1f}')
        axes[0, 0].legend()
        
        # 2. 最终帧数量分布
        axes[0, 1].hist(self.frame_stats['final_frames_per_seq'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Final SOI Frames per Sequence')
        axes[0, 1].set_xlabel('Number of Final Frames')
        axes[0, 1].set_ylabel('Number of Sequences')
        axes[0, 1].axvline(np.mean(self.frame_stats['final_frames_per_seq']), 
                          color='red', linestyle='--',
                          label=f'Mean: {np.mean(self.frame_stats["final_frames_per_seq"]):.1f}')
        axes[0, 1].legend()
        
        # 3. 候选帧vs最终帧对比
        axes[1, 0].scatter(self.frame_stats['candidates_per_seq'], 
                          self.frame_stats['final_frames_per_seq'], alpha=0.6)
        axes[1, 0].plot([0, max(self.frame_stats['candidates_per_seq'])], 
                       [0, max(self.frame_stats['candidates_per_seq'])], 
                       'r--', alpha=0.5, label='y=x')
        axes[1, 0].set_title('Candidate vs Final Frames per Sequence')
        axes[1, 0].set_xlabel('Candidate Frames')
        axes[1, 0].set_ylabel('Final Frames')
        axes[1, 0].legend()
        
        # 4. 过滤原因分布
        if self.frame_stats['filter_reasons']:
            reasons = list(self.frame_stats['filter_reasons'].keys())
            counts = list(self.frame_stats['filter_reasons'].values())
            
            # 创建饼图
            wedges, texts, autotexts = axes[1, 1].pie(counts, labels=reasons, autopct='%1.1f%%')
            axes[1, 1].set_title('Frame Filtering Reasons Distribution')
            
            # 美化饼图文字
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No filtering reasons data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Frame Filtering Reasons Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step3_2_frame_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成统计报告
        self._generate_frame_report(output_dir)
    
    def _generate_box_report(self, output_dir: str):
        """生成框统计报告"""
        report_path = os.path.join(output_dir, 'step3_1_box_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Step 3.1: Box Filtering Statistics Report ===\n\n")
            
            # 基本统计
            f.write("📊 Basic Statistics:\n")
            f.write(f"Total frames processed: {len(self.box_stats['soi_boxes_per_frame'])}\n")
            f.write(f"Total sequences: {len(set(self.box_stats['sequences']))}\n\n")
            
            # SOI框统计
            soi_boxes = self.box_stats['soi_boxes_per_frame']
            f.write("🎯 SOI Boxes Statistics:\n")
            f.write(f"Mean SOI boxes per frame: {np.mean(soi_boxes):.2f}\n")
            f.write(f"Std SOI boxes per frame: {np.std(soi_boxes):.2f}\n")
            f.write(f"Max SOI boxes in single frame: {max(soi_boxes)}\n")
            f.write(f"Frames with 0 SOI boxes: {soi_boxes.count(0)} ({soi_boxes.count(0)/len(soi_boxes)*100:.1f}%)\n\n")
            
            # 检测框统计
            det_boxes = self.box_stats['det_boxes_per_frame']
            f.write("🔍 Detection Boxes Statistics:\n")
            f.write(f"Mean detection boxes per frame: {np.mean(det_boxes):.2f}\n")
            f.write(f"Std detection boxes per frame: {np.std(det_boxes):.2f}\n")
            f.write(f"Max detection boxes in single frame: {max(det_boxes)}\n")
            f.write(f"Frames with 0 detection boxes: {det_boxes.count(0)} ({det_boxes.count(0)/len(det_boxes)*100:.1f}%)\n\n")
            
            # 过滤效果
            total_before = sum(self.box_stats['total_boxes_per_frame'])
            total_after = sum(self.box_stats['filtered_boxes_per_frame'])
            f.write("✂️ Filtering Effect:\n")
            f.write(f"Total boxes before filtering: {total_before:,}\n")
            f.write(f"Total boxes after filtering: {total_after:,}\n")
            f.write(f"Boxes filtered out: {total_before - total_after:,}\n")
            f.write(f"Filtering ratio: {(total_before - total_after)/total_before*100:.2f}%\n")
    
    def _generate_frame_report(self, output_dir: str):
        """生成帧统计报告"""
        report_path = os.path.join(output_dir, 'step3_2_frame_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Step 3.2: SOI Frame Extraction Statistics Report ===\n\n")
            
            # 基本统计
            f.write("📊 Basic Statistics:\n")
            f.write(f"Total sequences processed: {len(self.frame_stats['sequences'])}\n")
            f.write(f"Total candidate frames: {sum(self.frame_stats['candidates_per_seq'])}\n")
            f.write(f"Total final SOI frames: {sum(self.frame_stats['final_frames_per_seq'])}\n\n")
            
            # 序列级统计
            candidates = self.frame_stats['candidates_per_seq']
            final_frames = self.frame_stats['final_frames_per_seq']
            
            f.write("🎯 Per-Sequence Statistics:\n")
            f.write(f"Mean candidate frames per sequence: {np.mean(candidates):.1f}\n")
            f.write(f"Mean final frames per sequence: {np.mean(final_frames):.1f}\n")
            f.write(f"Sequences with 0 candidates: {candidates.count(0)}\n")
            f.write(f"Sequences with 0 final frames: {final_frames.count(0)}\n\n")
            
            # 过滤效果
            total_candidates = sum(candidates)
            total_final = sum(final_frames)
            if total_candidates > 0:
                retention_rate = total_final / total_candidates
                f.write("✂️ Filtering Effect:\n")
                f.write(f"Frame retention rate: {retention_rate*100:.2f}%\n")
                f.write(f"Frames filtered out: {total_candidates - total_final}\n\n")
            
            # 过滤原因
            if self.frame_stats['filter_reasons']:
                f.write("🔍 Filtering Reasons:\n")
                total_filtered = sum(self.frame_stats['filter_reasons'].values())
                for reason, count in sorted(self.frame_stats['filter_reasons'].items(), 
                                           key=lambda x: x[1], reverse=True):
                    f.write(f"{reason}: {count} ({count/total_filtered*100:.1f}%)\n")


def run_step3_box_filtering(config, dataset) -> str:
    """运行Step 3.1: 边界框过滤和合并(按序列跳过) - 增强版with统计"""
    print("🔄 Starting Step 3.1: Box filtering and merging (with statistics)")
    
    output_dir = os.path.join(config.output_dir, "step3_1_filtered_boxes")
    os.makedirs(output_dir, exist_ok=True)
    
    processor = DataProcessor(config)
    stats = Step3Statistics()
    
    for seq in tqdm(dataset, desc="Processing sequences"):
        seq_name = seq.name
        output_path = os.path.join(output_dir, f"{seq_name}.jsonl")

        if config.skip_existing_results and os.path.exists(output_path):
            if config.verbose:
                print(f"⏩ Skipped {seq_name}: already exists.")
            continue
        
        try:
            # 收集统计数据
            soi_counts = []
            det_counts = []
            total_counts = []
            filtered_counts = []
            
            for frame_idx in range(len(seq.ground_truth_rect)):
                # 收集原始框数量
                soi_boxes = processor.collect_soi_boxes(seq_name, frame_idx)
                det_boxes = processor.collect_detection_boxes(seq_name, frame_idx)
                
                soi_count = len(soi_boxes)
                det_count = len(det_boxes)
                total_count = soi_count + det_count
                
                soi_counts.append(soi_count)
                det_counts.append(det_count)
                total_counts.append(total_count)
            
            # 执行处理
            filtered_boxes = processor.process_sequence(seq_name, seq.dataset, seq.ground_truth_rect)
            
            # 收集过滤后的框数量
            for frame_boxes in filtered_boxes:
                filtered_counts.append(len(frame_boxes) - 1)  # 减去GT框
            
            # 添加统计数据
            stats.add_box_stats(seq_name, soi_counts, det_counts, total_counts, filtered_counts)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                for frame_boxes in filtered_boxes:
                    f.write(json.dumps(frame_boxes, ensure_ascii=False) + '\n')
            
            if config.verbose:
                avg_soi = np.mean(soi_counts)
                avg_det = np.mean(det_counts)
                avg_filtered = np.mean(filtered_counts)
                print(f"✅ {seq_name}: {len(filtered_boxes)} frames, "
                      f"avg SOI: {avg_soi:.1f}, avg DET: {avg_det:.1f}, avg filtered: {avg_filtered:.1f}")
                
        except Exception as e:
            print(f"❌ Failed to process {seq_name}: {e}")
            continue
    
    # 生成统计图表和报告
    if hasattr(config, 'generate_statistics') and config.generate_statistics:
        print("📊 Generating statistics...")
        stats.plot_box_statistics(output_dir)
        print(f"📈 Statistics saved to {output_dir}")
    
    print(f"✅ Step 3.1 completed. Results saved to {output_dir}")
    return output_dir


def run_step3_frame_extraction(config, dataset, step3_1_dir: str) -> str:
    """运行Step 3.2: SOI帧提取(按序列跳过) - 增强版with统计"""
    print("🔄 Starting Step 3.2: SOI frame extraction (with statistics)")
    
    output_dir = os.path.join(config.output_dir, "step3_2_soi_frames")
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = FrameExtractor(config)
    stats = Step3Statistics()
    
    for seq in tqdm(dataset, desc="Extracting SOI frames"):
        seq_name = seq.name
        step3_1_jsonl = os.path.join(step3_1_dir, f"{seq_name}.jsonl")
        output_path = os.path.join(output_dir, f"{seq_name}_soi_frames.jsonl")

        # 跳过已完成序列
        if config.skip_existing_results and os.path.exists(output_path):
            if config.verbose:
                print(f"⏩ Skipped {seq_name}: already exists.")
            continue
        
        if not os.path.exists(step3_1_jsonl):
            if config.verbose:
                print(f"⚠️ Warning: No filtered boxes found for {seq_name}")
            continue

        try:
            # 先获取候选帧（用于统计）
            trackers = extractor.load_tracker_status(seq_name, seq.dataset)
            if not trackers:
                continue
                
            candidates = extractor.extract_soi_candidates(trackers, len(seq.frames), seq.ground_truth_rect)
            
            # 提取最终SOI帧（包含详细过滤统计）
            soi_frames = extractor.extract_soi_frames(
                seq_name=seq_name,
                dataset_name=seq.dataset,
                frames=seq.frames,
                ground_truth_rects=seq.ground_truth_rect,
                step3_1_jsonl=step3_1_jsonl
            )
            
            # 计算过滤原因（简化版本，实际可以在extractor中详细追踪）
            filter_reasons = {}
            filtered_count = len(candidates) - len(soi_frames)
            if filtered_count > 0:
                # 这里可以添加更详细的过滤原因统计
                filter_reasons['quality_or_gap_filter'] = filtered_count
            
            # 添加统计数据
            stats.add_frame_stats(seq_name, len(candidates), len(soi_frames), filter_reasons)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(soi_frames, f, ensure_ascii=False)
            
            if config.verbose:
                print(f"✅ {seq_name}: {len(candidates)} candidates → {len(soi_frames)} final SOI frames")
                
        except Exception as e:
            print(f"❌ Failed to extract SOI frames for {seq_name}: {e}")
            continue
    
    # 生成统计图表和报告
    if hasattr(config, 'generate_statistics') and config.generate_statistics:
        print("📊 Generating statistics...")
        stats.plot_frame_statistics(output_dir)
        print(f"📈 Statistics saved to {output_dir}")
    
    print(f"✅ Step 3.2 completed. Results saved to {output_dir}")
    return output_dir