# pytracking/soi_pipeline/utils/analysis.py
import os
import json
import pandas as pd
from typing import Dict, List, Any


def analyze_soi_frame_statistics(soi_frames_dir: str) -> pd.DataFrame:
    """分析SOI帧统计信息"""
    stats = []
    
    if not os.path.exists(soi_frames_dir):
        print(f"Warning: SOI frames directory not found: {soi_frames_dir}")
        return pd.DataFrame()
    
    for file in os.listdir(soi_frames_dir):
        if not file.endswith("_soi_frames.jsonl"):
            continue
        
        seq_name = file.replace("_soi_frames.jsonl", "")
        file_path = os.path.join(soi_frames_dir, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                frame_indices = json.load(f)
            
            stats.append({
                "sequence": seq_name,
                "num_soi_frames": len(frame_indices),
                "min_frame_idx": min(frame_indices) if frame_indices else None,
                "max_frame_idx": max(frame_indices) if frame_indices else None,
                "frame_indices": frame_indices
            })
        except Exception as e:
            print(f"Warning: Failed to analyze {file}: {e}")
            continue
    
    if not stats:
        print("No SOI frame data found")
        return pd.DataFrame()
    
    df = pd.DataFrame(stats)
    
    # 打印统计信息
    print("📊 SOI Frame Statistics:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Average SOI frames per sequence: {df['num_soi_frames'].mean():.2f}")
    print(f"  Max SOI frames: {df['num_soi_frames'].max()}")
    print(f"  Min SOI frames: {df['num_soi_frames'].min()}")
    print(f"  Total SOI frames: {df['num_soi_frames'].sum()}")
    
    return df


def analyze_description_quality(descriptions_dir: str) -> Dict[str, Any]:
    """分析VLM描述质量"""
    total_descriptions = 0
    successful_descriptions = 0
    failed_descriptions = 0
    empty_descriptions = 0
    
    if not os.path.exists(descriptions_dir):
        print(f"Warning: Descriptions directory not found: {descriptions_dir}")
        return {}
    
    for file in os.listdir(descriptions_dir):
        if not file.endswith("_descriptions.jsonl"):
            continue
        
        file_path = os.path.join(descriptions_dir, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line.strip())
                    total_descriptions += 1
                    
                    vlm_output = data.get("vlm_output", "").strip()
                    
                    if not vlm_output:
                        empty_descriptions += 1
                        continue
                    
                    try:
                        # 尝试解析为JSON
                        parsed = json.loads(vlm_output)
                        if isinstance(parsed, dict) and len(parsed) >= 4:
                            # 检查是否包含必要的level
                            required_levels = ["level1", "level2", "level3", "level4"]
                            if all(level in parsed for level in required_levels):
                                successful_descriptions += 1
                            else:
                                failed_descriptions += 1
                        else:
                            failed_descriptions += 1
                    except json.JSONDecodeError:
                        failed_descriptions += 1
                        
        except Exception as e:
            print(f"Warning: Failed to analyze {file}: {e}")
            continue
    
    if total_descriptions == 0:
        print("No description data found")
        return {}
    
    success_rate = successful_descriptions / total_descriptions
    
    metrics = {
        "total_descriptions": total_descriptions,
        "successful_descriptions": successful_descriptions,
        "failed_descriptions": failed_descriptions,
        "empty_descriptions": empty_descriptions,
        "success_rate": success_rate
    }
    
    print("📊 Description Quality Metrics:")
    for key, value in metrics.items():
        if key == "success_rate":
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    return metrics


def generate_analysis_report(output_dir: str) -> None:
    """生成完整的分析报告"""
    print("🔍 Generating analysis report...")
    
    # 分析SOI帧
    soi_frames_dir = os.path.join(output_dir, "step3_2_soi_frames")
    soi_stats = analyze_soi_frame_statistics(soi_frames_dir)
    
    # 保存SOI帧统计
    if not soi_stats.empty:
        stats_file = os.path.join(output_dir, "soi_frame_statistics.csv")
        soi_stats.to_csv(stats_file, index=False)
        print(f"✅ SOI frame statistics saved to {stats_file}")
    
    # 分析描述质量
    descriptions_dir = os.path.join(output_dir, "step4_vlm_descriptions")
    quality_metrics = analyze_description_quality(descriptions_dir)
    
    # 保存质量指标
    if quality_metrics:
        metrics_file = os.path.join(output_dir, "description_quality_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(quality_metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ Quality metrics saved to {metrics_file}")
    
    print("📋 Analysis report generation completed")