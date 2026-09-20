import json
import os
import numpy as np
from pathlib import Path
from PIL import Image

def main():
    packet_dir = Path("data/tamil_ocr_dataset/imported/tamil/train/packet_001")
    manifest = packet_dir / "manifest.jsonl"
    
    widths = []
    prop_widths = []
    
    with open(manifest, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            img_path = packet_dir / entry["image"]
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    prop_width = round((w * 32) / h)
                    prop_widths.append(prop_width)
            except:
                pass
                
    widths = np.array(widths)
    prop_widths = np.array(prop_widths)
    
    report = {
        "original_width": {
            "min": int(np.min(widths)),
            "max": int(np.max(widths)),
            "mean": float(np.mean(widths)),
            "median": float(np.median(widths)),
            "p90": float(np.percentile(widths, 90)),
            "p95": float(np.percentile(widths, 95)),
            "p99": float(np.percentile(widths, 99))
        },
        "proportional_width_at_h32": {
            "min": int(np.min(prop_widths)),
            "max": int(np.max(prop_widths)),
            "mean": float(np.mean(prop_widths)),
            "median": float(np.median(prop_widths)),
            "p90": float(np.percentile(prop_widths, 90)),
            "p95": float(np.percentile(prop_widths, 95)),
            "p99": float(np.percentile(prop_widths, 99))
        },
        "exceeds": {
            "128": int(np.sum(prop_widths > 128)),
            "256": int(np.sum(prop_widths > 256)),
            "384": int(np.sum(prop_widths > 384)),
            "512": int(np.sum(prop_widths > 512)),
            "768": int(np.sum(prop_widths > 768)),
            "1024": int(np.sum(prop_widths > 1024)),
            "1536": int(np.sum(prop_widths > 1536)),
            "2048": int(np.sum(prop_widths > 2048))
        }
    }
    
    out_dir = Path("outputs/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resized_width_distribution.json", "w") as f:
        json.dump(report, f, indent=4)
        
    md_content = f"""# Resized Width Distribution

## Original Width Statistics
- Minimum: {report['original_width']['min']}
- Maximum: {report['original_width']['max']}
- Mean: {report['original_width']['mean']:.2f}
- Median: {report['original_width']['median']:.2f}
- P90: {report['original_width']['p90']:.2f}
- P95: {report['original_width']['p95']:.2f}
- P99: {report['original_width']['p99']:.2f}

## Proportional Width Statistics (at Height 32)
- Minimum: {report['proportional_width_at_h32']['min']}
- Maximum: {report['proportional_width_at_h32']['max']}
- Mean: {report['proportional_width_at_h32']['mean']:.2f}
- Median: {report['proportional_width_at_h32']['median']:.2f}
- P90: {report['proportional_width_at_h32']['p90']:.2f}
- P95: {report['proportional_width_at_h32']['p95']:.2f}
- P99: {report['proportional_width_at_h32']['p99']:.2f}

## Exceedance Counts
Number of images where proportional width exceeds:
- 128: {report['exceeds']['128']}
- 256: {report['exceeds']['256']}
- 384: {report['exceeds']['384']}
- 512: {report['exceeds']['512']}
- 768: {report['exceeds']['768']}
- 1024: {report['exceeds']['1024']}
- 1536: {report['exceeds']['1536']}
- 2048: {report['exceeds']['2048']}
"""
    report_dir = Path("docs/reports")
    with open(report_dir / "resized_width_distribution.md", "w") as f:
        f.write(md_content)

if __name__ == "__main__":
    main()
