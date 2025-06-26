# pytracking/soi_pipeline/scripts/run_step5_visualization.py
#!/usr/bin/env python3
"""
Step 5 Visualization Tool
Generates comprehensive visualizations and analysis for Step 5 verification results.
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from ..utils.step5_visualization import run_step5_visualization

def main():
    parser = argparse.ArgumentParser(description="Enhanced VLM-Guided Tracking Analysis Tool")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                       help="Directory containing Step 5 verification JSONL result files")
    parser.add_argument("--output_dir", "-o", type=str, default="./step5_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--iou_threshold", "-t", type=float, default=0.5,
                       help="IoU threshold for success classification (default: 0.5)")
    
    args = parser.parse_args()
    
    run_step5_visualization(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold
    )

if __name__ == "__main__":
    main()

"""
python -m pytracking.soi_pipeline.scripts.run_step5_visualization \
  --input_dir ./soi_outputs/step5_verification_results \
  --output_dir ./soi_outputs/step5_analysis
"""