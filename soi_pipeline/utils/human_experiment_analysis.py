# pytracking/soi_pipeline/analysis/human_experiment_analysis.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import scipy.stats as stats
from pathlib import Path

from .verification_utils import (
    load_jsonl, save_jsonl, compute_iou, xywh_to_xyxy
)

class HumanExperimentAnalyzer:
    """人类实验结果分析器"""
    
    def __init__(self, results_file: str, output_dir: str):
        self.results_file = results_file
        self.output_dir = output_dir
        self.analysis_dir = os.path.join(output_dir, "analysis_results")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # 加载数据
        self.human_results = self.load_human_results()
        print(f"✅ Loaded {len(self.human_results)} human annotation results")
        
        # 分析配置
        self.iou_threshold = 0.5  # 成功判定的IoU阈值
        self.confidence_levels = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
        self.difficulty_levels = {1: "Very Easy", 2: "Easy", 3: "Medium", 4: "Difficult", 5: "Very Difficult"}
        self.semantic_levels = {
            1: "Visual Only",
            2: "Visual + Position", 
            3: "Visual + Position + Appearance",
            4: "Visual + Position + Appearance + Dynamics",
            5: "All Information",
            6: "Cannot Determine"
        }
    
    def load_human_results(self) -> List[Dict]:
        """加载人类标注结果"""
        if not os.path.exists(self.results_file):
            print(f"❌ Results file not found: {self.results_file}")
            return []
        
        results = load_jsonl(self.results_file)
        
        # 过滤有效结果
        valid_results = []
        for result in results:
            if result.get('status') == 'completed' and result.get('human_results'):
                valid_results.append(result)
        
        return valid_results
    
    def compute_human_metrics(self) -> Dict:
        """计算人类标注的性能指标"""
        print("📊 Computing human performance metrics...")
        
        metrics = {
            "total_samples": len(self.human_results),
            "successful_annotations": 0,
            "average_iou": 0,
            "average_confidence": 0,
            "average_difficulty": 0,
            "semantic_level_usage": Counter(),
            "confidence_distribution": Counter(),
            "difficulty_distribution": Counter(),
            "success_by_semantic_level": defaultdict(lambda: {"total": 0, "success": 0, "ious": []}),
            "iou_distribution": [],
            "time_statistics": {
                "average_time": 0,
                "median_time": 0,
                "min_time": float('inf'),
                "max_time": 0
            }
        }
        
        ious = []
        confidences = []
        difficulties = []
        times = []
        
        for result in self.human_results:
            human_data = result['human_results']
            
            # 跳过无法确定的情况
            if human_data.get('selected_level') == 6:
                continue
            
            # 计算IoU
            human_bbox = human_data.get('bounding_box')
            gt_bbox = result.get('ground_truth_box')
            
            if human_bbox and gt_bbox:
                # 转换为xyxy格式
                human_xyxy = xywh_to_xyxy(human_bbox)
                gt_xyxy = xywh_to_xyxy(gt_bbox)
                
                if human_xyxy and gt_xyxy:
                    iou = compute_iou(gt_xyxy, human_xyxy)
                    ious.append(iou)
                    
                    # 判断是否成功
                    is_success = iou >= self.iou_threshold
                    if is_success:
                        metrics["successful_annotations"] += 1
                    
                    # 按语义层级统计
                    level = human_data.get('selected_level', 1)
                    metrics["success_by_semantic_level"][level]["total"] += 1
                    metrics["success_by_semantic_level"][level]["ious"].append(iou)
                    if is_success:
                        metrics["success_by_semantic_level"][level]["success"] += 1
            
            # 收集其他统计数据
            if human_data.get('confidence'):
                confidences.append(human_data['confidence'])
                metrics["confidence_distribution"][human_data['confidence']] += 1
            
            if human_data.get('difficulty'):
                difficulties.append(human_data['difficulty'])
                metrics["difficulty_distribution"][human_data['difficulty']] += 1
            
            if human_data.get('selected_level'):
                metrics["semantic_level_usage"][human_data['selected_level']] += 1
            
            if human_data.get('time_spent'):
                times.append(human_data['time_spent'])
        
        # 计算统计指标
        if ious:
            metrics["average_iou"] = np.mean(ious)
            metrics["iou_distribution"] = ious
        
        if confidences:
            metrics["average_confidence"] = np.mean(confidences)
        
        if difficulties:
            metrics["average_difficulty"] = np.mean(difficulties)
        
        if times:
            metrics["time_statistics"]["average_time"] = np.mean(times)
            metrics["time_statistics"]["median_time"] = np.median(times)
            metrics["time_statistics"]["min_time"] = min(times)
            metrics["time_statistics"]["max_time"] = max(times)
        
        return metrics
    
    def compare_human_vs_vlm(self) -> Dict:
        """对比人类与VLM的性能"""
        print("🔍 Comparing human vs VLM performance...")
        
        comparison = {
            "human_metrics": {},
            "vlm_metrics": {},
            "comparative_analysis": {},
            "statistical_tests": {}
        }
        
        # 提取对比数据
        human_ious = []
        vlm_ious_by_level = {"12": [], "123": [], "1234": []}
        human_success = []
        vlm_success_by_level = {"12": [], "123": [], "1234": []}
        
        for result in self.human_results:
            human_data = result['human_results']
            vlm_data = result.get('vlm_results', {})
            
            # 跳过无法确定的情况
            if human_data.get('selected_level') == 6:
                continue
            
            # 人类数据
            human_bbox = human_data.get('bounding_box')
            gt_bbox = result.get('ground_truth_box')
            
            if human_bbox and gt_bbox:
                human_xyxy = xywh_to_xyxy(human_bbox)
                gt_xyxy = xywh_to_xyxy(gt_bbox)
                
                if human_xyxy and gt_xyxy:
                    human_iou = compute_iou(gt_xyxy, human_xyxy)
                    human_ious.append(human_iou)
                    human_success.append(human_iou >= self.iou_threshold)
                    
                    # VLM数据
                    for level in ["12", "123", "1234"]:
                        vlm_iou = vlm_data.get(f'iou_{level}', 0)
                        vlm_ious_by_level[level].append(vlm_iou)
                        vlm_success_by_level[level].append(vlm_data.get(f'success_{level}', False))
        
        # 计算对比指标
        if human_ious:
            comparison["human_metrics"] = {
                "mean_iou": np.mean(human_ious),
                "std_iou": np.std(human_ious),
                "success_rate": np.mean(human_success),
                "median_iou": np.median(human_ious)
            }
            
            for level in ["12", "123", "1234"]:
                if vlm_ious_by_level[level]:
                    comparison["vlm_metrics"][f"level_{level}"] = {
                        "mean_iou": np.mean(vlm_ious_by_level[level]),
                        "std_iou": np.std(vlm_ious_by_level[level]),
                        "success_rate": np.mean(vlm_success_by_level[level]),
                        "median_iou": np.median(vlm_ious_by_level[level])
                    }
                    
                    # 统计显著性检验
                    if len(human_ious) > 5 and len(vlm_ious_by_level[level]) > 5:
                        t_stat, p_value = stats.ttest_ind(human_ious, vlm_ious_by_level[level])
                        comparison["statistical_tests"][f"ttest_level_{level}"] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        }
        
        return comparison
    
    def identify_failure_cases(self) -> Dict:
        """识别人类也失败的案例"""
        print("🔍 Identifying human failure cases...")
        
        failure_analysis = {
            "human_failures": [],
            "mutual_failures": [],  # 人类和VLM都失败
            "human_only_failures": [],  # 只有人类失败
            "vlm_only_failures": [],  # 只有VLM失败
            "failure_patterns": {},
            "description_issues": []
        }
        
        for result in self.human_results:
            human_data = result['human_results']
            vlm_data = result.get('vlm_results', {})
            
            # 跳过无法确定的情况
            if human_data.get('selected_level') == 6:
                continue
            
            # 计算人类表现
            human_bbox = human_data.get('bounding_box')
            gt_bbox = result.get('ground_truth_box')
            
            if not (human_bbox and gt_bbox):
                continue
            
            human_xyxy = xywh_to_xyxy(human_bbox)
            gt_xyxy = xywh_to_xyxy(gt_bbox)
            
            if not (human_xyxy and gt_xyxy):
                continue
            
            human_iou = compute_iou(gt_xyxy, human_xyxy)
            human_success = human_iou >= self.iou_threshold
            
            # VLM表现（使用最高级别1234）
            vlm_success = vlm_data.get('success_1234', False)
            vlm_iou = vlm_data.get('iou_1234', 0)
            
            # 分类失败案例
            failure_case = {
                "experiment_id": result['experiment_id'],
                "sequence_name": result['sequence_name'],
                "frame_idx": result['frame_idx'],
                "human_iou": human_iou,
                "human_success": human_success,
                "vlm_iou": vlm_iou,
                "vlm_success": vlm_success,
                "human_confidence": human_data.get('confidence', 0),
                "human_difficulty": human_data.get('difficulty', 0),
                "semantic_level_used": human_data.get('selected_level', 0),
                "comments": human_data.get('comments', ''),
                "descriptions": {
                    "level_1": result.get('description_level_1', ''),
                    "level_2": result.get('description_level_2', ''),
                    "level_3": result.get('description_level_3', ''),
                    "level_4": result.get('description_level_4', '')
                }
            }
            
            if not human_success:
                failure_analysis["human_failures"].append(failure_case)
                
                if not vlm_success:
                    failure_analysis["mutual_failures"].append(failure_case)
                else:
                    failure_analysis["human_only_failures"].append(failure_case)
            elif not vlm_success:
                failure_analysis["vlm_only_failures"].append(failure_case)
        
        # 分析失败模式
        self._analyze_failure_patterns(failure_analysis)
        
        return failure_analysis
    
    def _analyze_failure_patterns(self, failure_analysis: Dict):
        """分析失败模式"""
        
        patterns = {
            "difficulty_correlation": {},
            "confidence_correlation": {},
            "semantic_level_effectiveness": {},
            "description_quality_issues": []
        }
        
        # 分析人类失败案例的特征
        for case in failure_analysis["human_failures"]:
            # 难度相关性
            difficulty = case["human_difficulty"]
            if difficulty not in patterns["difficulty_correlation"]:
                patterns["difficulty_correlation"][difficulty] = 0
            patterns["difficulty_correlation"][difficulty] += 1
            
            # 信心度相关性
            confidence = case["human_confidence"]
            if confidence not in patterns["confidence_correlation"]:
                patterns["confidence_correlation"][confidence] = 0
            patterns["confidence_correlation"][confidence] += 1
            
            # 语义层级效果
            level = case["semantic_level_used"]
            if level not in patterns["semantic_level_effectiveness"]:
                patterns["semantic_level_effectiveness"][level] = {"total": 0, "failed": 0}
            patterns["semantic_level_effectiveness"][level]["total"] += 1
            patterns["semantic_level_effectiveness"][level]["failed"] += 1
            
            # 检查描述质量问题
            issues = self._check_description_quality(case["descriptions"], case["comments"])
            if issues:
                patterns["description_quality_issues"].append({
                    "experiment_id": case["experiment_id"],
                    "issues": issues,
                    "comments": case["comments"]
                })
        
        failure_analysis["failure_patterns"] = patterns
    
    def _check_description_quality(self, descriptions: Dict, comments: str) -> List[str]:
        """检查描述质量问题"""
        issues = []
        
        for level, desc in descriptions.items():
            if not desc or desc.strip() == "":
                continue
            
            desc_lower = desc.lower()
            
            # 检查模糊表达
            vague_words = ["maybe", "possibly", "seems", "appears", "might", "could be", "perhaps"]
            if any(word in desc_lower for word in vague_words):
                issues.append(f"Level {level}: Contains vague language")
            
            # 检查过于简短
            if len(desc.split()) < 3:
                issues.append(f"Level {level}: Too brief")
            
            # 检查冗余表达
            if len(desc.split()) > 50:
                issues.append(f"Level {level}: Too verbose")
            
            # 检查矛盾表达
            contradictory_pairs = [
                ("left", "right"), ("top", "bottom"), ("large", "small"),
                ("fast", "slow"), ("bright", "dark")
            ]
            for word1, word2 in contradictory_pairs:
                if word1 in desc_lower and word2 in desc_lower:
                    issues.append(f"Level {level}: Contains contradictory terms")
        
        # 检查用户评论中的问题提示
        if comments:
            comments_lower = comments.lower()
            if any(word in comments_lower for word in ["unclear", "confusing", "ambiguous", "wrong"]):
                issues.append("User reported description issues")
        
        return issues
    
    def generate_analysis_report(self) -> str:
        """生成完整的分析报告"""
        print("📊 Generating comprehensive analysis report...")
        
        # 计算各项指标
        human_metrics = self.compute_human_metrics()
        comparison = self.compare_human_vs_vlm()
        failure_analysis = self.identify_failure_cases()
        
        # 生成可视化图表
        self.generate_visualization_plots(human_metrics, comparison, failure_analysis)
        
        # 保存分析数据
        analysis_data = {
            "human_metrics": human_metrics,
            "human_vs_vlm_comparison": comparison,
            "failure_analysis": failure_analysis,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
        
        analysis_file = os.path.join(self.analysis_dir, "complete_analysis.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成markdown报告
        report_content = self._generate_markdown_report(human_metrics, comparison, failure_analysis)
        report_file = os.path.join(self.analysis_dir, "analysis_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Analysis report saved to: {report_file}")
        return report_file
    
    def _generate_markdown_report(self, human_metrics: Dict, comparison: Dict, failure_analysis: Dict) -> str:
        """生成Markdown格式的分析报告"""
        
        report = f"""# Human-Machine Cognitive Difference Analysis Report

*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary

This report analyzes the cognitive differences between humans and Vision-Language Models (VLMs) in visual object tracking tasks, based on {human_metrics['total_samples']} annotated samples.

### Key Findings

- **Human Success Rate**: {human_metrics['successful_annotations'] / max(human_metrics['total_samples'], 1) * 100:.1f}%
- **Average Human IoU**: {human_metrics['average_iou']:.3f}
- **Human vs VLM Performance**: {'Humans outperform VLMs' if human_metrics['average_iou'] > comparison.get('vlm_metrics', {}).get('level_1234', {}).get('mean_iou', 0) else 'VLMs show competitive performance'}
- **Mutual Failure Cases**: {len(failure_analysis['mutual_failures'])} cases where both humans and VLMs failed

## 1. Human Performance Analysis

### 1.1 Overall Metrics
- **Total Samples**: {human_metrics['total_samples']}
- **Successful Annotations**: {human_metrics['successful_annotations']}
- **Average IoU**: {human_metrics['average_iou']:.3f}
- **Average Confidence**: {human_metrics['average_confidence']:.2f}/5
- **Average Difficulty Rating**: {human_metrics['average_difficulty']:.2f}/5

### 1.2 Semantic Level Usage
"""
        
        # 添加语义层级使用统计
        for level, count in human_metrics['semantic_level_usage'].items():
            percentage = count / human_metrics['total_samples'] * 100
            level_name = self.semantic_levels.get(level, f"Level {level}")
            report += f"- **{level_name}**: {count} uses ({percentage:.1f}%)\n"
        
        report += f"""
### 1.3 Performance by Semantic Level
"""
        
        for level, stats in human_metrics['success_by_semantic_level'].items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total'] * 100
                avg_iou = np.mean(stats['ious'])
                level_name = self.semantic_levels.get(level, f"Level {level}")
                report += f"- **{level_name}**: {success_rate:.1f}% success rate, {avg_iou:.3f} average IoU\n"
        
        report += f"""
## 2. Human vs VLM Comparison

### 2.1 Performance Comparison
"""
        
        if 'human_metrics' in comparison and 'vlm_metrics' in comparison:
            human_perf = comparison['human_metrics']
            report += f"- **Human Performance**: {human_perf['success_rate']*100:.1f}% success rate, {human_perf['mean_iou']:.3f} mean IoU\n"
            
            for level_key, vlm_perf in comparison['vlm_metrics'].items():
                level = level_key.replace('level_', '')
                report += f"- **VLM Level {level}**: {vlm_perf['success_rate']*100:.1f}% success rate, {vlm_perf['mean_iou']:.3f} mean IoU\n"
        
        report += f"""
### 2.2 Statistical Significance
"""
        
        for test_key, test_result in comparison.get('statistical_tests', {}).items():
            level = test_key.replace('ttest_level_', '')
            significance = "statistically significant" if test_result['significant'] else "not statistically significant"
            report += f"- **Level {level}**: Difference is {significance} (p={test_result['p_value']:.4f})\n"
        
        report += f"""
## 3. Failure Analysis

### 3.1 Failure Case Distribution
- **Human-only failures**: {len(failure_analysis['human_only_failures'])} cases
- **VLM-only failures**: {len(failure_analysis['vlm_only_failures'])} cases  
- **Mutual failures**: {len(failure_analysis['mutual_failures'])} cases

### 3.2 Failure Patterns
"""
        
        patterns = failure_analysis.get('failure_patterns', {})
        
        # 难度分析
        if 'difficulty_correlation' in patterns:
            report += "#### Difficulty Correlation\n"
            for difficulty, count in patterns['difficulty_correlation'].items():
                diff_name = self.difficulty_levels.get(difficulty, f"Level {difficulty}")
                report += f"- **{diff_name}**: {count} failure cases\n"
        
        # 描述质量问题
        if 'description_quality_issues' in patterns:
            report += f"\n#### Description Quality Issues\n"
            report += f"- **Total cases with issues**: {len(patterns['description_quality_issues'])}\n"
            
            issue_types = Counter()
            for case in patterns['description_quality_issues']:
                for issue in case['issues']:
                    issue_type = issue.split(':')[1].strip() if ':' in issue else issue
                    issue_types[issue_type] += 1
            
            for issue_type, count in issue_types.most_common():
                report += f"- **{issue_type}**: {count} cases\n"
        
        report += f"""
## 4. Recommendations

### 4.1 For VLM Improvement
"""
        
        # 基于分析结果生成建议
        if human_metrics['average_iou'] > comparison.get('vlm_metrics', {}).get('level_1234', {}).get('mean_iou', 0):
            report += "- Humans demonstrate superior performance, suggesting VLMs need better semantic understanding\n"
            report += "- Focus on improving spatial reasoning and context understanding capabilities\n"
        
        if len(failure_analysis['mutual_failures']) > 0:
            report += f"- {len(failure_analysis['mutual_failures'])} cases challenge both humans and VLMs - consider improving description quality\n"
        
        report += """
### 4.2 For Description Generation
"""
        
        if patterns.get('description_quality_issues'):
            report += "- Reduce vague and ambiguous language in descriptions\n"
            report += "- Ensure consistency across semantic levels\n"
            report += "- Balance detail level - avoid both oversimplification and verbosity\n"
        
        report += f"""
### 4.3 For Future Research
- Investigate the {len(failure_analysis['mutual_failures'])} mutual failure cases for fundamental limitations
- Develop better metrics for semantic description quality
- Explore multi-modal approaches combining visual and textual information

## 5. Conclusion

This study reveals important insights into human-machine cognitive differences in visual object tracking. The findings suggest {'human superiority in semantic understanding' if human_metrics['average_iou'] > 0.6 else 'comparable performance with room for improvement in both human and machine capabilities'}.

---
*Report generated by SOI Pipeline Step 6 Analysis Tool*
"""
        
        return report
    
    def generate_visualization_plots(self, human_metrics: Dict, comparison: Dict, failure_analysis: Dict):
        """生成可视化图表"""
        print("📊 Generating visualization plots...")
        
        plots_dir = os.path.join(self.analysis_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 人类vs VLM性能对比
        self._plot_human_vs_vlm_comparison(comparison, plots_dir)
        
        # 2. 语义层级使用分布
        self._plot_semantic_level_usage(human_metrics, plots_dir)
        
        # 3. IoU分布直方图
        self._plot_iou_distribution(human_metrics, plots_dir)
        
        # 4. 失败案例分析
        self._plot_failure_analysis(failure_analysis, plots_dir)
        
        # 5. 信心度与表现关系
        self._plot_confidence_performance(plots_dir)
        
        print(f"✅ Visualization plots saved to: {plots_dir}")
    
    def _plot_human_vs_vlm_comparison(self, comparison: Dict, plots_dir: str):
        """绘制人机性能对比图"""
        if not comparison.get('human_metrics') or not comparison.get('vlm_metrics'):
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 成功率对比
        human_success = comparison['human_metrics']['success_rate']
        vlm_success_rates = [comparison['vlm_metrics'][f'level_{level}']['success_rate'] 
                           for level in ['12', '123', '1234'] 
                           if f'level_{level}' in comparison['vlm_metrics']]
        
        categories = ['Human'] + [f'VLM L{level}' for level in ['12', '123', '1234'][:len(vlm_success_rates)]]
        success_rates = [human_success] + vlm_success_rates
        
        bars1 = ax1.bar(categories, success_rates, alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax1.set_ylabel('Success Rate', fontsize=12)
        ax1.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # IoU对比
        human_iou = comparison['human_metrics']['mean_iou']
        vlm_ious = [comparison['vlm_metrics'][f'level_{level}']['mean_iou'] 
                   for level in ['12', '123', '1234'] 
                   if f'level_{level}' in comparison['vlm_metrics']]
        
        iou_values = [human_iou] + vlm_ious
        
        bars2 = ax2.bar(categories, iou_values, alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax2.set_ylabel('Average IoU', fontsize=12)
        ax2.set_title('Average IoU Comparison', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for bar, iou in zip(bars2, iou_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{iou:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "human_vs_vlm_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_semantic_level_usage(self, human_metrics: Dict, plots_dir: str):
        """绘制语义层级使用分布图"""
        if not human_metrics.get('semantic_level_usage'):
            return
        
        plt.figure(figsize=(12, 8))
        
        levels = list(human_metrics['semantic_level_usage'].keys())
        counts = list(human_metrics['semantic_level_usage'].values())
        labels = [self.semantic_levels.get(level, f"Level {level}") for level in levels]
        
        # 饼图
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Semantic Information Level Usage Distribution', 
                 fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "semantic_level_usage.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_iou_distribution(self, human_metrics: Dict, plots_dir: str):
        """绘制IoU分布直方图"""
        if not human_metrics.get('iou_distribution'):
            return
        
        plt.figure(figsize=(12, 6))
        
        ious = human_metrics['iou_distribution']
        plt.hist(ious, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        plt.xlabel('IoU Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Human Annotation IoU Distribution', fontsize=16, fontweight='bold')
        
        # 添加统计线
        plt.axvline(np.mean(ious), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(ious):.3f}')
        plt.axvline(np.median(ious), color='green', linestyle='--', 
                   label=f'Median: {np.median(ious):.3f}')
        plt.axvline(self.iou_threshold, color='orange', linestyle='--', 
                   label=f'Success Threshold: {self.iou_threshold}')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "iou_distribution.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_failure_analysis(self, failure_analysis: Dict, plots_dir: str):
        """绘制失败案例分析图"""
        # 失败类型分布
        plt.figure(figsize=(10, 6))
        
        failure_types = ['Human Only', 'VLM Only', 'Mutual Failures']
        failure_counts = [
            len(failure_analysis['human_only_failures']),
            len(failure_analysis['vlm_only_failures']),
            len(failure_analysis['mutual_failures'])
        ]
        
        bars = plt.bar(failure_types, failure_counts, alpha=0.8, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.ylabel('Number of Cases', fontsize=12)
        plt.title('Failure Case Distribution', fontsize=16, fontweight='bold')
        
        # 添加数值标签
        for bar, count in zip(bars, failure_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "failure_distribution.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_performance(self, plots_dir: str):
        """绘制信心度与表现关系图"""
        confidence_performance = defaultdict(list)
        
        for result in self.human_results:
            human_data = result['human_results']
            
            if human_data.get('selected_level') == 6:  # 跳过无法确定
                continue
            
            confidence = human_data.get('confidence')
            if not confidence:
                continue
            
            # 计算IoU
            human_bbox = human_data.get('bounding_box')
            gt_bbox = result.get('ground_truth_box')
            
            if human_bbox and gt_bbox:
                human_xyxy = xywh_to_xyxy(human_bbox)
                gt_xyxy = xywh_to_xyxy(gt_bbox)
                
                if human_xyxy and gt_xyxy:
                    iou = compute_iou(gt_xyxy, human_xyxy)
                    confidence_performance[confidence].append(iou)
        
        if confidence_performance:
            plt.figure(figsize=(10, 6))
            
            confidences = sorted(confidence_performance.keys())
            avg_ious = [np.mean(confidence_performance[conf]) for conf in confidences]
            conf_labels = [self.confidence_levels[conf] for conf in confidences]
            
            plt.bar(conf_labels, avg_ious, alpha=0.8, color='lightcoral')
            plt.ylabel('Average IoU', fontsize=12)
            plt.xlabel('Confidence Level', fontsize=12)
            plt.title('Confidence vs Performance Relationship', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "confidence_performance.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def run_human_experiment_analysis(results_file: str, output_dir: str) -> str:
    """运行人类实验分析"""
    print("🔄 Starting human experiment analysis...")
    
    analyzer = HumanExperimentAnalyzer(results_file, output_dir)
    report_file = analyzer.generate_analysis_report()
    
    print("✅ Human experiment analysis completed!")
    return report_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Human Experiment Analysis Tool")
    parser.add_argument("--results-file", required=True, help="Path to human annotation results file")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis")
    
    args = parser.parse_args()
    
    run_human_experiment_analysis(args.results_file, args.output_dir)