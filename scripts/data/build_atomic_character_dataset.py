import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/TamilNet_old/TamilChar.csv")
    parser.add_argument("--mapping", type=str, default="outputs/fast_track/ATOMIC_CHARACTER_MAPPING.csv")
    parser.add_argument("--vocab", type=str, default="data/tamil_ocr_dataset/vocabulary/tamil_vocab.json")
    parser.add_argument("--out", type=str, default="data/atomic_chars_synthetic")
    parser.add_argument("--augment-count", type=int, default=10)
    args = parser.parse_args()

    print("=== ATOMIC CHARACTER DATASET BUILDER (SMOKE TEST) ===")
    print("WARNING: TamilChar.csv labels are opaque integers (0-155).")
    print("WARNING: No Unicode mapping is provided.")
    print("ERROR: Cannot map opaque integers to the 77-class atomic vocabulary.")
    print("Status: ABORTED. External dataset with explicit Unicode labels is required.")

if __name__ == "__main__":
    main()
