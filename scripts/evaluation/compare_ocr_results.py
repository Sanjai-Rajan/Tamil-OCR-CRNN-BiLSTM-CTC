import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON")
    parser.add_argument("--new", required=True, help="New metrics JSON")
    args = parser.parse_args()

    b_data = json.load(open(args.baseline))
    n_data = json.load(open(args.new))

    print(f"{'Metric':<25} | {'Baseline':<10} | {'New':<10} | {'Improvement'}")
    print("-" * 65)

    metrics = ["CER", "WER", "Character Accuracy", "Word Accuracy", "Blank Percentage"]
    for m in metrics:
        b_val = b_data.get(m, 0)
        n_val = n_data.get(m, 0)
        
        if "Accuracy" in m:
            diff = n_val - b_val
            imp = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        else:
            diff = b_val - n_val
            imp = f"-{-diff:.4f}" if diff > 0 else f"+{-diff:.4f}"
            
        print(f"{m:<25} | {b_val:<10.4f} | {n_val:<10.4f} | {imp}")

if __name__ == "__main__":
    main()
