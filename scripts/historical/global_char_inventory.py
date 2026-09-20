import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import csv
from pathlib import Path
from collections import Counter

def get_vocab():
    vocab_path = Path("data/tamil_ocr_dataset/vocabulary/tamil_vocab.json")
    if not vocab_path.exists():
        return set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return set(data.get("characters", []))

def main():
    roots = [Path("data/tamil_ocr_dataset/imported")]
    vocab = get_vocab()
    
    char_counter = Counter()
    dataset_occurrences = {}
    
    # Scan all manifest.jsonl
    for root in roots:
        for r, d, f in os.walk(root):
            if "manifest.jsonl" in f:
                manifest_path = Path(r) / "manifest.jsonl"
                dataset_name = manifest_path.parent.name
                
                with open(manifest_path, 'r', encoding='utf-8') as mf:
                    for line in mf:
                        if not line.strip(): continue
                        try:
                            entry = json.loads(line)
                            label = entry.get('label', '')
                            for char in label:
                                char_counter[char] += 1
                                if char not in dataset_occurrences:
                                    dataset_occurrences[char] = set()
                                dataset_occurrences[char].add(dataset_name)
                        except:
                            pass
                            
    out_dir = Path("outputs/pre_5090_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_data = []
    for char, count in char_counter.items():
        unicode_hex = f"U+{ord(char):04X}"
        in_vocab = "Yes" if char in vocab else "No"
        datasets = "|".join(sorted(list(dataset_occurrences[char])))
        
        csv_data.append({
            "character": char,
            "Unicode codepoints": unicode_hex,
            "dataset occurrences": datasets,
            "frequency": count,
            "current vocabulary membership": in_vocab,
            "normalization status": "N/A"
        })
        
    csv_data.sort(key=lambda x: x["frequency"], reverse=True)
    
    with open(out_dir / "GLOBAL_CHARACTER_INVENTORY.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["character", "Unicode codepoints", "dataset occurrences", "frequency", "current vocabulary membership", "normalization status"])
        writer.writeheader()
        writer.writerows(csv_data)
        
if __name__ == "__main__":
    main()
