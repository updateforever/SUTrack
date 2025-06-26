# pytracking/soi_pipeline/scripts/analyze_results.py
#!/usr/bin/env python3
"""
分析SOI Pipeline结果的独立脚本
Usage: python soi_pipeline/scripts/analyze_results.py --output-dir ./soi_outputs
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from soi_pipeline.utils.analysis import generate_analysis_report


def main():
    parser = argparse.ArgumentParser(description="Analyze SOI Pipeline Results")
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='SOI Pipeline output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Error: Output directory not found: {args.output_dir}")
        return
    
    print(f"🔍 Analyzing results in: {args.output_dir}")
    generate_analysis_report(args.output_dir)


if __name__ == "__main__":
    main()