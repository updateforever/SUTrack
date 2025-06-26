# pytracking/soi_pipeline/main.py
#!/usr/bin/env python3
"""
SOI Pipeline Main Entry Point
Usage: python -m soi_pipeline.main [args]
"""

import os
import sys
import argparse
from typing import List

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入PyTracking数据集加载器
from lib.test.evaluation import get_dataset

# 导入SOI Pipeline模块
from .configs.config import Config, BatchConfig, HumanExperimentConfig
from .pipelines.step3_pipeline import run_step3_box_filtering, run_step3_frame_extraction
from .pipelines.step4_pipeline import run_step4_description_generation, validate_step4_output
from .pipelines.step5_pipeline import run_step5_pipeline, VerificationConfig, validate_step5_output
from .pipelines.step6_pipeline import run_step6_preparation

from .utils.analysis import generate_analysis_report


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SOI Pipeline for PyTracking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with cleaning and verification
  python -m soi_pipeline.main --dataset lasot --soi-tracker-dir /path/to/soi --api-key your-key

  # Run specific steps
  python -m soi_pipeline.main --steps 3 --dataset lasot --soi-tracker-dir /path/to/soi
  python -m soi_pipeline.main --steps 4 5 --api-key your-key

  # Run step 4 with text cleaning disabled
  python -m soi_pipeline.main --steps 4 --no-cleaning --api-key your-key

  # Run step 5 verification only
  python -m soi_pipeline.main --steps 5 --use-local-model --model-path /path/to/model

  # Run step 6 with custom batching
  python -m soi_pipeline.main --steps 6 --batch-size 150 --stratify-by iou_range --create-overlap

  # Use local model
  python -m soi_pipeline.main --use-local-model --model-path /path/to/qwen2.5-vl
        """
    )
    
    # 基本配置
    parser.add_argument('--dataset', type=str, default='lasot', 
                       help='Dataset name (default: lasot)')
    parser.add_argument('--output-dir', type=str, default='./soi_outputs', 
                       help='Output directory (default: ./soi_outputs)')
    parser.add_argument('--steps', type=int, nargs='+', choices=[3, 4, 5, 6], default=[3, 4, 5, 6], 
                       help='Pipeline steps to run (default: 3 4 5 6)')
    
    # Step 3 配置
    parser.add_argument('--soi-tracker-dir', type=str, 
                       help='Directory containing SOI tracker results')
    parser.add_argument('--detection-dir', type=str, 
                       help='Directory containing detection results')
    parser.add_argument('--status-dir', type=str, 
                       help='Directory containing tracker status files')
    
    # VLM配置
    parser.add_argument('--use-local-model', action='store_true', default=True, 
                       help='Use local VLM model instead of API')
    parser.add_argument('--model-path', type=str, default='/mnt/first/wangyipei/qwenvl32b', 
                       help='Path to local VLM model')
    parser.add_argument('--api-key', type=str, 
                       help='API key for VLM service')
    parser.add_argument('--use-template', action='store_true',  
                       help='Use template image for description generation')
    
    # 处理参数
    parser.add_argument('--iou-thresh-gt', type=float, default=0.5, 
                       help='IoU threshold for GT filtering (default: 0.5)')
    parser.add_argument('--iou-thresh-nms', type=float, default=0.4, 
                       help='IoU threshold for NMS (default: 0.4)')
    parser.add_argument('--frame-gap-threshold', type=int, default=30, 
                       help='Minimum frame gap between SOI frames (default: 30)')
    parser.add_argument('--min-vote-ratio', type=float, default=0.5, 
                       help='Minimum vote ratio for SOI frame selection (default: 0.5)')
    
    # Step 4 特定参数（描述生成+文本清理）
    parser.add_argument('--no-cleaning', action='store_true', 
                       help='Disable text cleaning in step 4')
    parser.add_argument('--small-area-thresh', type=float, default=0.001,
                       help='Small area threshold for filtering boxes (default: 0.001)')
    parser.add_argument('--min-box-size', type=int, default=10,
                       help='Minimum box size in pixels (default: 10)')
    
    # Step 5 特定参数（仅反向验证）
    parser.add_argument('--verification-ref-mode', type=str, default='first', 
                       choices=['none', 'first', 'prev'],
                       help='Reference frame mode for verification (default: prev)')
    parser.add_argument('--verification-levels', type=str, default='12,123,1234', 
                       help='Verification levels comma-separated (default: 12,123,1234)')
    parser.add_argument('--verification-iou-thresh', type=float, default=0.25, 
                       help='IoU threshold for verification success (default: 0.25)')
    parser.add_argument('--save-verification-vis', action='store_true', 
                       help='Save verification visualization images')
    
    # Step 6 批次配置参数
    parser.add_argument('--batch-size', type=int, default=200,
                       help='Number of samples per batch (default: 200)')
    parser.add_argument('--stratify-by', type=str, default='sequence_name',
                       choices=['sequence_name', 'iou_range', 'failure_type', 'none'],
                       help='Stratification strategy for batching (default: sequence_name)')
    parser.add_argument('--min-batch-size', type=int, default=50,
                       help='Minimum batch size (default: 50)')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Do not shuffle samples before batching')
    parser.add_argument('--create-overlap', action='store_true',
                       help='Create overlap validation batches')
    parser.add_argument('--overlap-ratio', type=float, default=0.1,
                       help='Overlap ratio for validation batches (default: 0.1)')
    parser.add_argument('--exclude-edge-cases', action='store_true',
                    help='Exclude edge cases near IoU threshold')
    parser.add_argument('--balance-classes', action='store_true', default=True,
                    help='Balance class distribution in batches')
    parser.add_argument('--random-seed', type=int, default=42,
                    help='Random seed for reproducible batching')

    # Step 6 实验配置参数
    parser.add_argument('--vlm-failure-threshold', type=float, default=0.3,
                       help='IoU threshold for VLM failure detection (default: 0.3)')
    parser.add_argument('--min-samples-per-seq', type=int, default=1,
                       help='Minimum samples per sequence (default: 1)')
    parser.add_argument('--max-samples-per-seq', type=int, default=2000,
                       help='Maximum samples per sequence (default: 20)')
    
    # Step 6 标注工具参数
    parser.add_argument('--run-annotation-tool', action='store_true',
                       help='Launch the annotation tool after data preparation')
    parser.add_argument('--annotation-batch-id', type=int,
                       help='Specific batch ID to annotate (if not specified, shows batch selection)')
    parser.add_argument('--gradio-port', type=int, default=7860,
                       help='Port for Gradio annotation interface (default: 7860)')
    parser.add_argument('--gradio-share', action='store_true',
                       help='Create shareable Gradio link')
    
    # 批次管理命令
    parser.add_argument('--list-batches', action='store_true',
                       help='List all available batches and their status')
    parser.add_argument('--batch-stats', action='store_true',
                       help='Show detailed statistics for all batches')
    parser.add_argument('--merge-results', action='store_true',
                       help='Merge completed batch results into final dataset')

    # 验证和分析选项
    parser.add_argument('--validate-step4', action='store_true',
                       help='Validate Step 4 output integrity')
    parser.add_argument('--validate-step5', action='store_true',
                       help='Validate Step 5 output integrity')

    # 其他选项
    parser.add_argument('--verbose', action='store_true', default=True, 
                       help='Enable verbose output')
    parser.add_argument('--save-visualizations', action='store_true', 
                       help='Save visualization images')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='Only run analysis on existing results')
    parser.add_argument('--no-skip', dest='skip_existing_results', action='store_false',
                    help='Do not skip sequences with existing results')
    
    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """从命令行参数创建配置对象"""
    # 解析验证级别
    verification_levels = args.verification_levels.split(',') if args.verification_levels else None
    
    return Config(
        dataset_name=args.dataset,
        soi_tracker_dir=args.soi_tracker_dir or '',
        detection_dir=args.detection_dir or '',
        status_dir=args.status_dir or '',
        output_dir=args.output_dir,
        
        iou_thresh_gt=args.iou_thresh_gt,
        iou_thresh_nms=args.iou_thresh_nms,
        frame_gap_threshold=args.frame_gap_threshold,
        min_vote_ratio=args.min_vote_ratio,
        
        use_local_model=args.use_local_model,
        model_path=args.model_path or '',
        api_key=args.api_key or '',
        use_template=args.use_template,
        
        # Step 4相关配置
        enable_text_cleaning=not args.no_cleaning,
        small_area_thresh=args.small_area_thresh,
        min_box_size=args.min_box_size,
        
        # Step 5相关配置  
        verification_ref_mode=args.verification_ref_mode,
        verification_levels=verification_levels,
        verification_iou_threshold=args.verification_iou_thresh,
        save_verification_visualizations=args.save_verification_vis,
        
        verbose=args.verbose,
        save_visualizations=args.save_visualizations
    )


def create_batch_config_from_args(args) -> BatchConfig:
    """从命令行参数创建批次配置"""
    return BatchConfig(
        batch_size=args.batch_size,
        stratify_by=args.stratify_by if args.stratify_by != 'none' else 'none',
        min_batch_size=args.min_batch_size,
        shuffle_samples=not args.no_shuffle,
        create_overlap=args.create_overlap,
        overlap_ratio=args.overlap_ratio,
        exclude_edge_cases=args.exclude_edge_cases,
        balance_classes=args.balance_classes,
        random_seed=args.random_seed
    )


def create_human_experiment_config_from_args(args) -> HumanExperimentConfig:
    """从命令行参数创建人类实验配置"""
    return HumanExperimentConfig(
        vlm_failure_threshold=args.vlm_failure_threshold,
        min_samples_per_sequence=args.min_samples_per_seq,
        max_samples_per_sequence=args.max_samples_per_seq,
        enable_confidence_rating=True,
        enable_difficulty_rating=True
    )


def validate_config(config: Config, steps: List[int]) -> bool:
    """验证配置有效性"""
    if 3 in steps:
        if not config.soi_tracker_dir:
            print("❌ Error: --soi-tracker-dir is required for Step 3")
            return False
        if not os.path.exists(config.soi_tracker_dir):
            print(f"❌ Error: SOI tracker directory not found: {config.soi_tracker_dir}")
            return False
    
    if 4 in steps:
        # Step 4需要VLM用于描述生成，可能需要API用于文本清理
        if config.use_local_model:
            if not config.model_path:
                print("❌ Error: --model-path is required for local model")
                return False
            if not os.path.exists(config.model_path):
                print(f"❌ Error: Model path not found: {config.model_path}")
                return False
        else:
            if not config.api_key:
                print("❌ Error: --api-key is required for API mode")
                return False
        
        # 如果启用文本清理，还需要API key
        if config.enable_text_cleaning and not config.api_key:
            print("❌ Error: --api-key is required for text cleaning")
            return False
    
    if 5 in steps:
        # Step 5需要VLM用于反向验证
        if config.use_local_model:
            if not config.model_path:
                print("❌ Error: --model-path is required for local model verification")
                return False
            if not os.path.exists(config.model_path):
                print(f"❌ Error: Model path not found: {config.model_path}")
                return False
        else:
            if not config.api_key:
                print("❌ Error: --api-key is required for API mode verification")
                return False
    
    return True


def list_batch_info(output_dir: str):
    """列出批次信息"""
    batches_dir = os.path.join(output_dir, "step6_human_experiment", "experiment_batches")
    
    if not os.path.exists(batches_dir):
        print("❌ No batches found. Please run Step 6 first.")
        return
    
    # 读取批次总览
    summary_file = os.path.join(batches_dir, "batches_summary.json")
    if os.path.exists(summary_file):
        import json
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"📊 Batch Summary")
        print(f"   Total batches: {summary['total_batches']}")
        print(f"   Total samples: {summary['total_samples']}")
        print(f"   Created: {summary['created_date']}")
        print()
        
        print("📋 Batch Details:")
        for batch_info in summary['batches']:
            batch_id = batch_info['batch_id']
            filename = batch_info['filename']
            sample_count = batch_info['sample_count']
            est_time = batch_info['estimated_time_minutes']
            sequences = len(batch_info['sequences'])
            
            # 检查是否已完成
            result_file = os.path.join(batches_dir, f"batch_{batch_id:03d}_results.jsonl")
            status = "✅ Completed" if os.path.exists(result_file) else "⏳ Pending"
            
            print(f"   Batch {batch_id:2d}: {sample_count:3d} samples, {sequences:2d} sequences, "
                  f"{est_time:4.0f}min - {status}")
    else:
        print("❌ Batch summary not found")


def show_batch_statistics(output_dir: str):
    """显示详细批次统计"""
    batches_dir = os.path.join(output_dir, "step6_human_experiment", "experiment_batches")
    
    if not os.path.exists(batches_dir):
        print("❌ No batches found. Please run Step 6 first.")
        return
    
    import json
    import glob
    
    print("📊 Detailed Batch Statistics")
    print("=" * 80)
    
    # 读取所有批次统计文件
    stats_files = glob.glob(os.path.join(batches_dir, "batch_*_stats.json"))
    stats_files.sort()
    
    for stats_file in stats_files:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        batch_id = stats['batch_id']
        print(f"\n🔹 Batch {batch_id}")
        print(f"   Samples: {stats['total_samples']}")
        print(f"   Sequences: {len(stats['sequences'])} ({', '.join(stats['sequences'][:5])}{'...' if len(stats['sequences']) > 5 else ''})")
        print(f"   IoU range: {stats['iou_range']['min']:.3f} - {stats['iou_range']['max']:.3f} (avg: {stats['iou_range']['avg']:.3f})")
        
        if stats['failure_types']:
            failure_summary = ", ".join([f"{k}: {v}" for k, v in list(stats['failure_types'].items())[:3]])
            print(f"   Failure types: {failure_summary}")


def merge_batch_results(output_dir: str):
    """合并批次结果"""
    batches_dir = os.path.join(output_dir, "step6_human_experiment", "experiment_batches")
    
    if not os.path.exists(batches_dir):
        print("❌ No batches found. Please run Step 6 first.")
        return
    
    import json
    import glob
    from .utils.verification_utils import load_jsonl, save_jsonl
    
    # 查找所有结果文件
    result_files = glob.glob(os.path.join(batches_dir, "batch_*_results.jsonl"))
    
    if not result_files:
        print("❌ No completed batch results found.")
        return
    
    print(f"🔄 Merging {len(result_files)} batch results...")
    
    all_results = []
    batch_summary = {}
    
    for result_file in sorted(result_files):
        batch_id = int(result_file.split('_')[-2])
        results = load_jsonl(result_file)
        
        completed_results = [r for r in results if r.get('status') == 'completed']
        all_results.extend(completed_results)
        
        batch_summary[batch_id] = {
            "total_samples": len(results),
            "completed_samples": len(completed_results),
            "completion_rate": len(completed_results) / len(results) if results else 0
        }
        
        print(f"   Batch {batch_id}: {len(completed_results)}/{len(results)} completed")
    
    # 保存合并结果
    merged_file = os.path.join(output_dir, "step6_human_experiment", "merged_human_annotations.jsonl")
    save_jsonl(merged_file, all_results)
    
    # 保存合并统计
    merge_stats = {
        "total_batches_processed": len(result_files),
        "total_samples": sum(s["total_samples"] for s in batch_summary.values()),
        "total_completed": len(all_results),
        "overall_completion_rate": len(all_results) / sum(s["total_samples"] for s in batch_summary.values()),
        "batch_summary": batch_summary,
        "merged_date": __import__('datetime').datetime.now().isoformat()
    }
    
    stats_file = os.path.join(output_dir, "step6_human_experiment", "merge_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(merge_stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Merged results saved to: {merged_file}")
    print(f"📊 Merge statistics saved to: {stats_file}")
    print(f"📈 Overall completion: {len(all_results)}/{merge_stats['total_samples']} "
          f"({merge_stats['overall_completion_rate']:.1%})")


def launch_batch_annotation_tool(output_dir: str, batch_id: int = None, 
                                gradio_port: int = 7860, gradio_share: bool = False):
    """启动批次标注工具"""
    batches_dir = os.path.join(output_dir, "step6_human_experiment", "experiment_batches")
    
    if not os.path.exists(batches_dir):
        print("❌ No batches found. Please run Step 6 first.")
        return
    
    # 如果未指定批次ID，显示选择界面
    if batch_id is None:
        print("📋 Available batches:")
        list_batch_info(output_dir)
        
        try:
            batch_id = int(input("\nEnter batch ID to annotate: "))
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid batch ID or cancelled.")
            return
    
    # 查找批次文件
    batch_file = os.path.join(batches_dir, f"batch_{batch_id:03d}_experiment_data.jsonl")
    
    if not os.path.exists(batch_file):
        print(f"❌ Batch {batch_id} not found: {batch_file}")
        return
    
    print(f"🚀 Launching annotation tool for Batch {batch_id}...")
    print(f"📁 Batch file: {batch_file}")
    print(f"🌐 Server will start on port {gradio_port}")
    
    # 这里需要导入并启动标注工具
    try:
        from .tools.gradio_annotation_tool import launch_batch_annotation_tool as launch_tool
        
        launch_tool(
            batch_file=batch_file,
            output_dir=batches_dir,
            batch_id=batch_id,
            server_port=gradio_port,
            share=gradio_share
        )
    except ImportError:
        print("❌ Annotation tool not available. Please implement gradio_annotation_tool.py")
    except Exception as e:
        print(f"❌ Failed to launch annotation tool: {e}")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 验证选项
    if args.validate_step4:
        step4_dir = os.path.join(config.output_dir, "step4_vlm_descriptions")
        validate_step4_output(step4_dir, verbose=args.verbose)
        return
    
    if args.validate_step5:
        step5_dir = os.path.join(config.output_dir, "step5_verification_results")
        validate_step5_output(step5_dir, verbose=args.verbose)
        return
    
    # 批次管理命令
    if args.list_batches:
        list_batch_info(config.output_dir)
        return
    
    if args.batch_stats:
        show_batch_statistics(config.output_dir)
        return
    
    if args.merge_results:
        merge_batch_results(config.output_dir)
        return
    
    if args.run_annotation_tool:
        launch_batch_annotation_tool(
            config.output_dir, 
            args.annotation_batch_id,
            args.gradio_port,
            args.gradio_share
        )
        return
    
    print("🚀 SOI Pipeline Starting...")
    print(f"📁 Output directory: {config.output_dir}")
    print(f"🎯 Steps to run: {args.steps}")
    
    # 仅分析模式
    if args.analyze_only:
        print("📊 Running analysis only...")
        generate_analysis_report(config.output_dir)
        return
    
    # 验证配置
    if not validate_config(config, args.steps):
        sys.exit(1)
    
    # 加载数据集
    dataset = None
    if 3 in args.steps or 4 in args.steps or 5 in args.steps or 6 in args.steps:
        try:
            print(f"🔄 Loading dataset: {config.dataset_name}")
            dataset = get_dataset(config.dataset_name)
            print(f"✅ Dataset loaded successfully: {len(dataset)} sequences")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            sys.exit(1)

    # 运行管道步骤
    step3_1_dir = None
    step3_2_dir = None
    step4_dir = None
    step5_dir = None

    try:
        # Step 3: 框过滤和SOI帧提取
        if 3 in args.steps:
            print("\n" + "="*60)
            print("STEP 3: Box Filtering and SOI Frame Extraction")
            print("="*60)
            
            step3_1_dir = run_step3_box_filtering(config, dataset)
            step3_2_dir = run_step3_frame_extraction(config, dataset, step3_1_dir)
        
        # Step 4: VLM描述生成 + 文本清理
        if 4 in args.steps:
            print("\n" + "="*60)
            print("STEP 4: VLM Description Generation and Text Cleaning")
            print("="*60)
            
            # 使用现有结果或Step 3输出
            if step3_1_dir is None:
                step3_1_dir = os.path.join(config.output_dir, "step3_1_filtered_boxes")
            if step3_2_dir is None:
                step3_2_dir = os.path.join(config.output_dir, "step3_2_soi_frames")
            
            if not os.path.exists(step3_1_dir) or not os.path.exists(step3_2_dir):
                print("❌ Error: Step 3 outputs not found. Please run Step 3 first.")
                sys.exit(1)
            
            # 显示Step 4配置
            print(f"📋 Step 4 Configuration:")
            print(f"   - Description generation: Enabled")
            print(f"   - Text cleaning: {'Enabled' if config.enable_text_cleaning else 'Disabled'}")
            print(f"   - Use template: {'Yes' if config.use_template else 'No'}")
            print(f"   - VLM model: {'Local' if config.use_local_model else 'API'}")
            if config.use_local_model:
                print(f"   - Model path: {config.model_path}")
            
            step4_dir = run_step4_description_generation(config, dataset, step3_1_dir, step3_2_dir)
            
            # 验证Step 4输出
            if args.verbose:
                print(f"\n🔍 Validating Step 4 output...")
                validate_step4_output(step4_dir, verbose=True)
        
        # Step 5: 反向验证
        if 5 in args.steps:
            print("\n" + "="*60)
            print("STEP 5: Reverse Verification")
            print("="*60)
            
            # 使用现有结果或Step 4输出
            if step4_dir is None:
                step4_dir = os.path.join(config.output_dir, "step4_vlm_descriptions")
            
            if not os.path.exists(step4_dir):
                print("❌ Error: Step 4 outputs not found. Please run Step 4 first.")
                sys.exit(1)
            
            # 显示Step 5配置
            print(f"📋 Step 5 Configuration:")
            print(f"   - Reference mode: {config.verification_ref_mode}")
            print(f"   - Verification levels: {config.verification_levels}")
            print(f"   - IoU threshold: {config.verification_iou_threshold}")
            print(f"   - Save visualizations: {'Yes' if config.save_verification_visualizations else 'No'}")
            
            # 创建验证配置
            verification_config = VerificationConfig(
                enable_verification=True,
                ref_mode=config.verification_ref_mode,
                levels=config.verification_levels,
                iou_threshold=config.verification_iou_threshold,
                save_visualizations=config.save_verification_visualizations
            )
            
            # 运行Step 5
            step5_results = run_step5_pipeline(
                config, step4_dir, dataset,
                verification_config=verification_config,
                run_analysis=True
            )
            
            # 显示结果
            if "verification_dir" in step5_results:
                print(f"✅ Verification results: {step5_results['verification_dir']}")
                step5_dir = step5_results["verification_dir"]
                
                # 验证Step 5输出
                if args.verbose:
                    print(f"\n🔍 Validating Step 5 output...")
                    validate_step5_output(step5_dir, verbose=True)
            
            if "analysis_dir" in step5_results:
                print(f"✅ Analysis results: {step5_results['analysis_dir']}")
        
        # Step 6: 人机语义认知能力差异探索（增强版）
        if 6 in args.steps:
            print("\n" + "="*60)
            print("STEP 6: Human-Machine Cognitive Difference Exploration (Enhanced)")
            print("="*60)
            
            # 使用现有结果或Step 5输出
            if step5_dir is None:
                step5_dir = os.path.join(config.output_dir, "step5_verification_results")
            
            if not os.path.exists(step5_dir):
                print("❌ Error: Step 5 verification outputs not found. Please run Step 5 first.")
                sys.exit(1)
            
            # 创建批次配置
            batch_config = create_batch_config_from_args(args)
            
            # 显示批次配置
            print(f"📦 Batch Configuration:")
            print(f"   - Batch size: {batch_config.batch_size}")
            print(f"   - Stratify by: {batch_config.stratify_by}")
            print(f"   - Min batch size: {batch_config.min_batch_size}")
            print(f"   - Shuffle samples: {batch_config.shuffle_samples}")
            print(f"   - Create overlap: {batch_config.create_overlap}")
            if batch_config.create_overlap:
                print(f"   - Overlap ratio: {batch_config.overlap_ratio}")
            
            # 创建人类实验配置
            human_experiment_config = create_human_experiment_config_from_args(args)
            
            # 显示实验配置
            print(f"🧪 Experiment Configuration:")
            print(f"   - VLM failure threshold: {human_experiment_config.vlm_failure_threshold}")
            print(f"   - Samples per sequence: {human_experiment_config.min_samples_per_sequence}-{human_experiment_config.max_samples_per_sequence}")
            
            # 运行Step 6
            step6_results = run_step6_preparation(
                config, step5_dir, dataset, 
                human_experiment_config, batch_config
            )
            
            # 显示结果
            print(f"✅ Step 6 Results:")
            print(f"   - Output directory: {step6_results['output_dir']}")
            print(f"   - Total batches: {step6_results['total_batches']}")
            print(f"   - Total samples: {step6_results['total_samples']}")
            print(f"   - Batches directory: {step6_results['batches_dir']}")
            
            # 显示后续步骤建议
            print(f"\n📋 Next Steps:")
            print(f"   1. Review batch assignment guide:")
            print(f"      {step6_results['batches_dir']}/batch_assignment_guide.md")
            print(f"   2. List available batches:")
            print(f"      python -m soi_pipeline.main --list-batches --output-dir {config.output_dir}")
            print(f"   3. Start annotation for a specific batch:")
            print(f"      python -m soi_pipeline.main --run-annotation-tool --annotation-batch-id 1")
            print(f"   4. Show detailed batch statistics:")
            print(f"      python -m soi_pipeline.main --batch-stats --output-dir {config.output_dir}")
            print(f"   5. Merge completed results:")
            print(f"      python -m soi_pipeline.main --merge-results --output-dir {config.output_dir}")
        
        # 生成分析报告（除非启动标注工具）
        if not args.run_annotation_tool:
            print("\n" + "="*60)
            print("ANALYSIS REPORT")
            print("="*60)
            generate_analysis_report(config.output_dir)
        
        print("\n🎉 SOI Pipeline completed successfully!")
        print(f"📁 All results saved to: {config.output_dir}")
        
        # 如果运行了Step 6，显示批次管理提示
        if 6 in args.steps:
            print(f"\n💡 Batch Management Tips:")
            print(f"   • Use --list-batches to see all available batches")
            print(f"   • Use --run-annotation-tool to start annotating")
            print(f"   • Use --merge-results when annotation is complete")
            print(f"   • Each batch is designed for ~{args.batch_size} samples")
        
        # 显示新的Step 4/5设计说明
        if 4 in args.steps or 5 in args.steps:
            print(f"\n📝 Pipeline Design Notes:")
            print(f"   • Step 4: Handles description generation + text cleaning")
            print(f"   • Step 5: Focuses solely on reverse verification")
            print(f"   • Use --validate-step4 or --validate-step5 to check output integrity")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        if config.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Enhanced Usage Examples with New Step 4/5 Design:

# 1. 运行完整流程（描述生成+清理+验证）
python -m soi_pipeline.main \
  --dataset lasot \
  --soi-tracker-dir /path/to/tracker/results \
  --api-key your-key \
  --use-local-model \
  --model-path /path/to/qwen2.5-vl

# 2. 运行Step 4（描述生成+清理）
python -m soi_pipeline.main --steps 4 \
  --soi-tracker-dir /home/wyp/project/SUTrack/soi/tracker_soi_results \
  --detection-dir /home/wyp/project/ultralytics/yoloworld_results/ \
  --output-dir ./soi_outputs \
  --status-dir /home/wyp/project/SUTrack/soi/trackers_status \
  --api-key xxx \
   

# 3. 运行Step 4但禁用文本清理
python -m soi_pipeline.main --steps 4 \
  --no-cleaning \
  --use-local-model \
  --model-path /path/to/qwen2.5-vl

# 4. 运行Step 5（仅反向验证）
python -m soi_pipeline.main --steps 5 \
  --use-local-model \
  --model-path /path/to/qwen2.5-vl \
  --verification-ref-mode prev \
  --verification-levels "12,123,1234"

# 5. 验证输出完整性
python -m soi_pipeline.main --validate-step4 --output-dir ./soi_outputs
python -m soi_pipeline.main --validate-step5 --output-dir ./soi_outputs

# 6. 运行特定步骤组合
python -m soi_pipeline.main --steps 4 5 \
  --api-key your-key \
  --use-local-model \
  --model-path /path/to/qwen2.5-vl \
  --save-verification-vis

# 7. 不同验证配置
python -m soi_pipeline.main --steps 5 \
  --verification-ref-mode first \
  --verification-levels "12,23,234" \
  --verification-iou-thresh 0.3

# 8. 完整的团队工作流
# Step 1: 数据准备（描述生成+清理）
python -m soi_pipeline.main --steps 3 4 \
  --soi-tracker-dir /path/to/tracker \
  --api-key your-key

# Step 2: 验证
python -m soi_pipeline.main --steps 5 \
  --use-local-model \
  --model-path /path/to/qwen2.5-vl

# Step 3: 批次准备
python -m soi_pipeline.main --steps 6 \
  --batch-size 200 \
  --stratify-by sequence_name

# Step 4: 验证输出
python -m soi_pipeline.main --validate-step4
python -m soi_pipeline.main --validate-step5

# Step 5: 开始标注
python -m soi_pipeline.main --run-annotation-tool
"""