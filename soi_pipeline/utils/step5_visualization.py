# pytracking/soi_pipeline/utils/step5_visualization.py
#!/usr/bin/env python3
"""
Step 5 Visualization Tool
Generates comprehensive visualizations and analysis for Step 5 verification results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_results(input_dir: str) -> pd.DataFrame:
    """Load all JSONL results from the input directory"""
    print("📂 Loading results from JSONL files...")
    records = []
    
    for fname in tqdm(os.listdir(input_dir)):
        if not fname.endswith(".jsonl"):
            continue
            
        fpath = os.path.join(input_dir, fname)
        sequence_name = fname.replace(".jsonl", "").replace("_verified", "")
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        # Extract data for all modes
                        record = {
                            "sequence": sequence_name,
                            "frame_idx": data.get("frame_idx", 0),
                        }
                        
                        # Add data for each description mode
                        for mode in ["12", "123", "1234"]:
                            record.update({
                                f"iou_{mode}": data.get(f"verification_iou_{mode}", 0),
                                f"ok_{mode}": data.get(f"verification_ok_{mode}", False),
                                f"pred_{mode}_exists": f"verification_pred_{mode}" in data
                            })
                        
                        records.append(record)
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error in {fname}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"❌ File not found: {fpath}")
            continue
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            continue
    
    if not records:
        print("❌ No valid records found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} records from {df['sequence'].nunique()} sequences")
    return df

def compute_summary_stats(df: pd.DataFrame, iou_threshold: float = 0.5) -> pd.DataFrame:
    """Compute summary statistics for each description mode"""
    print("📊 Computing summary statistics...")
    
    summary_data = []
    modes = ["12", "123", "1234"]
    
    for mode in modes:
        iou_col = f"iou_{mode}"
        ok_col = f"ok_{mode}"
        
        if iou_col in df.columns and ok_col in df.columns:
            mode_data = df[df[iou_col] > 0]  # Filter valid IoU values
            
            summary_data.append({
                "mode": mode,
                "accuracy": mode_data[ok_col].mean() if len(mode_data) > 0 else 0,
                "mean_iou": mode_data[iou_col].mean() if len(mode_data) > 0 else 0,
                "median_iou": mode_data[iou_col].median() if len(mode_data) > 0 else 0,
                "std_iou": mode_data[iou_col].std() if len(mode_data) > 0 else 0,
                "num_samples": len(mode_data),
                "success_rate": (mode_data[iou_col] >= iou_threshold).mean() if len(mode_data) > 0 else 0
            })
    
    return pd.DataFrame(summary_data)

def create_comprehensive_visualizations(df: pd.DataFrame, output_dir: str = "analysis_results") -> None:
    """Create comprehensive visualizations"""
    print("🎨 Creating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10
    })
    
    modes = ["12", "123", "1234"]
    mode_labels = ["Basic (1-2)", "Medium (1-2-3)", "Detailed (1-2-3-4)"]
    
    # 1. IoU Distribution Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('VLM-Guided Tracking Performance Analysis', fontsize=16, fontweight='bold')
    
    # IoU Box Plot
    iou_data = []
    for mode, label in zip(modes, mode_labels):
        iou_col = f"iou_{mode}"
        if iou_col in df.columns:
            valid_data = df[df[iou_col] > 0][iou_col]
            iou_data.extend([(label, val) for val in valid_data])
    
    if iou_data:
        iou_df = pd.DataFrame(iou_data, columns=['Mode', 'IoU'])
        sns.boxplot(data=iou_df, x='Mode', y='IoU', ax=axes[0,0])
        axes[0,0].set_title('IoU Distribution by Description Level', fontweight='bold')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Accuracy Bar Chart
    summary = compute_summary_stats(df)
    if not summary.empty:
        bars = axes[0,1].bar(mode_labels[:len(summary)], summary['accuracy'], 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        axes[0,1].set_title('Accuracy by Description Level', fontweight='bold')
        axes[0,1].set_ylabel('Accuracy (IoU ≥ 0.5)')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, summary['accuracy']):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Mean IoU Comparison
    if not summary.empty:
        bars = axes[1,0].bar(mode_labels[:len(summary)], summary['mean_iou'], 
                           color=['#FF9F43', '#6C5CE7', '#00B894'], alpha=0.8)
        axes[1,0].set_title('Mean IoU by Description Level', fontweight='bold')
        axes[1,0].set_ylabel('Mean IoU')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, iou in zip(bars, summary['mean_iou']):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{iou:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Sample Count
    if not summary.empty:
        bars = axes[1,1].bar(mode_labels[:len(summary)], summary['num_samples'], 
                           color=['#FD79A8', '#FDCB6E', '#6C5CE7'], alpha=0.8)
        axes[1,1].set_title('Number of Samples per Mode', fontweight='bold')
        axes[1,1].set_ylabel('Sample Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, summary['num_samples']):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(summary['num_samples']),
                         f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Per-Sequence Analysis
    create_sequence_analysis(df, output_dir)
    
    # 6. Performance Correlation Analysis
    create_correlation_analysis(df, output_dir)
    
    # 7. IoU Histogram
    create_iou_histograms(df, output_dir)

def create_sequence_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Create per-sequence performance analysis"""
    print("📈 Creating per-sequence analysis...")
    
    sequences = df['sequence'].unique()
    if len(sequences) == 0:
        return
    
    # Calculate per-sequence statistics
    seq_stats = []
    for seq in sequences:
        seq_data = df[df['sequence'] == seq]
        
        for mode in ["12", "123", "1234"]:
            iou_col = f"iou_{mode}"
            ok_col = f"ok_{mode}"
            
            if iou_col in seq_data.columns and ok_col in seq_data.columns:
                valid_data = seq_data[seq_data[iou_col] > 0]
                if len(valid_data) > 0:
                    seq_stats.append({
                        'sequence': seq,
                        'mode': mode,
                        'mean_iou': valid_data[iou_col].mean(),
                        'accuracy': valid_data[ok_col].mean(),
                        'frame_count': len(valid_data)
                    })
    
    if not seq_stats:
        return
    
    seq_df = pd.DataFrame(seq_stats)
    
    # Create heatmap for sequence performance
    if len(sequences) > 1:
        pivot_accuracy = seq_df.pivot(index='sequence', columns='mode', values='accuracy')
        pivot_iou = seq_df.pivot(index='sequence', columns='mode', values='mean_iou')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(sequences)*0.4)))
        
        # Accuracy heatmap
        sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax1, cbar_kws={'label': 'Accuracy'})
        ax1.set_title('Per-Sequence Accuracy Heatmap', fontweight='bold')
        ax1.set_xlabel('Description Mode')
        ax1.set_ylabel('Sequence')
        
        # Mean IoU heatmap
        sns.heatmap(pivot_iou, annot=True, fmt='.3f', cmap='viridis', 
                   ax=ax2, cbar_kws={'label': 'Mean IoU'})
        ax2.set_title('Per-Sequence Mean IoU Heatmap', fontweight='bold')
        ax2.set_xlabel('Description Mode')
        ax2.set_ylabel('Sequence')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sequence_heatmaps.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Top and bottom performing sequences
    if len(seq_df) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top performing sequences (by mean IoU across all modes)
        seq_overall = seq_df.groupby('sequence')['mean_iou'].mean().sort_values(ascending=False)
        top_n = min(10, len(seq_overall))
        
        top_sequences = seq_overall.head(top_n)
        bars1 = ax1.bar(range(len(top_sequences)), top_sequences.values, 
                       color='green', alpha=0.7)
        ax1.set_title(f'Top {top_n} Sequences by Mean IoU', fontweight='bold')
        ax1.set_ylabel('Mean IoU')
        ax1.set_xticks(range(len(top_sequences)))
        ax1.set_xticklabels(top_sequences.index, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars1, top_sequences.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Bottom performing sequences
        bottom_sequences = seq_overall.tail(top_n)
        bars2 = ax2.bar(range(len(bottom_sequences)), bottom_sequences.values, 
                       color='red', alpha=0.7)
        ax2.set_title(f'Bottom {top_n} Sequences by Mean IoU', fontweight='bold')
        ax2.set_ylabel('Mean IoU')
        ax2.set_xticks(range(len(bottom_sequences)))
        ax2.set_xticklabels(bottom_sequences.index, rotation=45, ha='right')
        ax2.set_ylim(0, max(0.5, bottom_sequences.max()))
        
        # Add value labels
        for bar, val in zip(bars2, bottom_sequences.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sequence_ranking.png"), dpi=300, bbox_inches='tight')
        plt.close()

def create_correlation_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Create correlation analysis between different modes"""
    print("🔗 Creating correlation analysis...")
    
    # Prepare correlation data
    corr_data = df[['iou_12', 'iou_123', 'iou_1234']].copy()
    corr_data = corr_data[corr_data > 0].dropna()  # Remove invalid IoU values
    
    if corr_data.empty:
        return
    
    # Correlation matrix
    correlation_matrix = corr_data.corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
               center=0, ax=ax1, square=True)
    ax1.set_title('IoU Correlation Between Description Modes', fontweight='bold')
    
    # Scatter plot matrix (correlation visualization)
    modes = ['iou_12', 'iou_123', 'iou_1234']
    colors = ['red', 'blue']
    
    # Plot IoU_12 vs IoU_123
    ax2.scatter(corr_data['iou_12'], corr_data['iou_123'], alpha=0.6, c='blue', label='12 vs 123')
    ax2.scatter(corr_data['iou_123'], corr_data['iou_1234'], alpha=0.6, c='red', label='123 vs 1234')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect correlation')
    ax2.set_xlabel('IoU (simpler mode)')
    ax2.set_ylabel('IoU (more detailed mode)')
    ax2.set_title('IoU Correlation Scatter Plot', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_iou_histograms(df: pd.DataFrame, output_dir: str) -> None:
    """Create IoU distribution histograms"""
    print("📊 Creating IoU histograms...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    modes = ["12", "123", "1234"]
    mode_labels = ["Basic (1-2)", "Medium (1-2-3)", "Detailed (1-2-3-4)"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (mode, label, color) in enumerate(zip(modes, mode_labels, colors)):
        iou_col = f"iou_{mode}"
        if iou_col in df.columns:
            valid_data = df[df[iou_col] > 0][iou_col]
            
            if len(valid_data) > 0:
                axes[i].hist(valid_data, bins=30, alpha=0.7, color=color, edgecolor='black')
                axes[i].axvline(valid_data.mean(), color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {valid_data.mean():.3f}')
                axes[i].axvline(0.5, color='orange', linestyle='-', linewidth=2, 
                              label='Success threshold (0.5)')
                axes[i].set_title(f'IoU Distribution - {label}', fontweight='bold')
                axes[i].set_xlabel('IoU')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou_histograms.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    """Save detailed results to CSV files"""
    print("💾 Saving detailed results...")
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path, index=False)
    
    # Save per-sequence statistics
    sequences = df['sequence'].unique()
    seq_results = []
    
    for seq in sequences:
        seq_data = df[df['sequence'] == seq]
        for mode in ["12", "123", "1234"]:
            iou_col = f"iou_{mode}"
            ok_col = f"ok_{mode}"
            
            if iou_col in seq_data.columns and ok_col in seq_data.columns:
                valid_data = seq_data[seq_data[iou_col] > 0]
                if len(valid_data) > 0:
                    seq_results.append({
                        'sequence': seq,
                        'mode': mode,
                        'mean_iou': valid_data[iou_col].mean(),
                        'median_iou': valid_data[iou_col].median(),
                        'std_iou': valid_data[iou_col].std(),
                        'accuracy': valid_data[ok_col].mean(),
                        'frame_count': len(valid_data),
                        'success_count': valid_data[ok_col].sum()
                    })
    
    if seq_results:
        seq_df = pd.DataFrame(seq_results)
        seq_path = os.path.join(output_dir, "per_sequence_results.csv")
        seq_df.to_csv(seq_path, index=False)
    
    # Save raw data (filtered)
    raw_path = os.path.join(output_dir, "raw_data_filtered.csv")
    df.to_csv(raw_path, index=False)
    
    print(f"✅ Results saved to {output_dir}/")

def run_step5_visualization(input_dir: str, output_dir: str, iou_threshold: float = 0.5) -> Dict:
    """
    Run Step 5 visualization and analysis
    
    Args:
        input_dir: Directory containing Step 5 verification results
        output_dir: Directory to save analysis results
        iou_threshold: IoU threshold for success classification
    
    Returns:
        Dictionary with summary results
    """
    print("🚀 Starting Enhanced VLM Tracking Analysis...")
    print(f"📂 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 IoU threshold: {iou_threshold}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_all_results(input_dir)
    if df.empty:
        print("❌ No data loaded. Exiting.")
        return {}
    
    # Compute summary statistics
    summary = compute_summary_stats(df, iou_threshold)
    
    # Create visualizations
    create_comprehensive_visualizations(df, output_dir)
    
    # Save results
    save_detailed_results(df, summary, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    
    summary_dict = {}
    
    if not summary.empty:
        for _, row in summary.iterrows():
            mode = row['mode']
            accuracy = row['accuracy']
            mean_iou = row['mean_iou']
            samples = int(row['num_samples'])
            
            print(f"Mode {mode:>4}: Accuracy={accuracy:.3f}, Mean IoU={mean_iou:.3f}, Samples={samples}")
            
            summary_dict[mode] = {
                'accuracy': accuracy,
                'mean_iou': mean_iou,
                'samples': samples
            }
    else:
        print("No summary statistics available.")
    
    print("\n✅ Analysis complete! Check the output directory for detailed results and visualizations.")
    
    return {
        'summary': summary_dict,
        'output_dir': output_dir,
        'total_samples': len(df),
        'sequences': df['sequence'].nunique()
    }