import os
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def main():
    out_dir = Path("outputs/diagnostics/training_length_bias_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_path = Path("outputs/diagnostics/training_length_distribution/training_length_buckets.csv")
    df_train = pd.read_csv(train_csv_path)
    total_train = int(df_train['sample_count'].sum())
    
    df_train.to_csv(out_dir / "training_distribution.csv", index=False)
    
    w4_test = {
        "1-5": {"samples": 33674, "target_avg": 1.97, "pred_avg": 1.99, "cer": 6.96, "word_acc": 87.30},
        "6-10": {"samples": 7358, "target_avg": 8.13, "pred_avg": 5.53, "cer": 45.34, "word_acc": 0.52},
        "11-15": {"samples": 1745, "target_avg": 12.57, "pred_avg": 5.98, "cer": 57.20, "word_acc": 0.0},
        "16+": {"samples": 333, "target_avg": 17.92, "pred_avg": 6.38, "cer": 66.63, "word_acc": 0.0},
    }
    total_test = 43110

    test_dist = []
    w4_metrics = []
    for b in ["1-5", "6-10", "11-15", "16+"]:
        cnt = int(w4_test[b]["samples"])
        pct = float(cnt / total_test * 100)
        test_dist.append({"bucket": b, "sample_count": cnt, "percentage": pct})
        w4_metrics.append({
            "bucket": b,
            "sample_count": cnt,
            "avg_target_length": float(w4_test[b]["target_avg"]),
            "avg_prediction_length": float(w4_test[b]["pred_avg"]),
            "pred_target_ratio": float(w4_test[b]["pred_avg"] / w4_test[b]["target_avg"]),
            "cer": float(w4_test[b]["cer"]),
            "word_accuracy": float(w4_test[b]["word_acc"])
        })
        
    df_test = pd.DataFrame(test_dist)
    df_test.to_csv(out_dir / "test_distribution.csv", index=False)
    
    df_w4 = pd.DataFrame(w4_metrics)
    df_w4.to_csv(out_dir / "length_wise_recognition_metrics.csv", index=False)
    
    comp = []
    for b in ["1-5", "6-10", "11-15", "16+"]:
        trn = df_train[df_train['bucket'] == b].iloc[0]
        tst = df_test[df_test['bucket'] == b].iloc[0]
        comp.append({
            "bucket": b,
            "train_samples": int(trn['sample_count']),
            "train_pct": float(trn['percentage']),
            "test_samples": int(tst['sample_count']),
            "test_pct": float(tst['percentage'])
        })
    df_comp = pd.DataFrame(comp)
    df_comp.to_csv(out_dir / "training_vs_test_comparison.csv", index=False)
    
    summary = {
        "total_train_samples": total_train,
        "total_test_samples": total_test,
        "max_train_length": int(df_train['max_target_length'].max()),
        "train_ge_20": int(df_train[df_train['bucket'] == '20+']['sample_count'].values[0]),
        "train_ge_25": int(df_train[df_train['bucket'] == '25+']['sample_count'].values[0]),
        "buckets": comp
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    plt.figure()
    x = range(4)
    w = 0.35
    plt.bar([i - w/2 for i in x], df_comp['train_pct'], w, label='Train %')
    plt.bar([i + w/2 for i in x], df_comp['test_pct'], w, label='Test %')
    plt.xticks(x, ["1-5", "6-10", "11-15", "16+"])
    plt.title("Train vs Test Length Distribution")
    plt.legend()
    plt.savefig(out_dir / "train_vs_test_distribution.png")
    
    plt.figure()
    plt.plot(x, df_w4['avg_target_length'], 'k--', label='Target')
    plt.plot(x, df_w4['avg_prediction_length'], 'ro-', label='Prediction')
    plt.xticks(x, ["1-5", "6-10", "11-15", "16+"])
    plt.title("W/4 Prediction Length Saturation")
    plt.legend()
    plt.savefig(out_dir / "w4_prediction_saturation.png")
    
    print("Diagnostic files generated successfully.")

if __name__ == '__main__':
    main()
