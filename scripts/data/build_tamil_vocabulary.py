import os
import json
from pathlib import Path
import argparse

def build_vocabulary(dataset_root, language):
    working_root = Path(dataset_root)
    train_dir = working_root / "imported" / language / "train"
    vocab_dir = working_root / "vocabulary"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_file = vocab_dir / f"{language}_vocab.json"
    
    unique_chars = set()
    total_samples = 0
    empty_labels = 0
    
    if not train_dir.exists():
        print(f"Error: Train directory {train_dir} does not exist. Import packets first.")
        return
        
    print(f"Scanning imported packets in {train_dir}...")
    for packet_dir in sorted(train_dir.iterdir()):
        if packet_dir.is_dir() and packet_dir.name.startswith("packet_"):
            manifest_path = packet_dir / "manifest.jsonl"
            if manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            label = entry.get("label", "")
                            total_samples += 1
                            if not label:
                                empty_labels += 1
                            for char in label:
                                unique_chars.add(char)
                                
    # Explicitly add space to support sentences/words even if training data lacks it currently
    unique_chars.add(" ")
    
    # Sort deterministically
    sorted_chars = sorted(list(unique_chars))
    
    # Build maps, keeping 0 for blank
    char_to_index = {"<blank>": 0}
    index_to_char = {"0": "<blank>"}
    
    for idx, char in enumerate(sorted_chars, start=1):
        char_to_index[char] = idx
        index_to_char[str(idx)] = char
        
    vocab_data = {
        "blank_index": 0,
        "total_samples_scanned": total_samples,
        "empty_labels": empty_labels,
        "unique_characters_count": len(sorted_chars),
        "characters": sorted_chars,
        "char_to_index": char_to_index,
        "index_to_char": index_to_char
    }
    
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=4)
        
    print(f"Vocabulary successfully built and saved to {vocab_file}")
    print(f"Total samples scanned: {total_samples}")
    print(f"Empty labels found in source: {empty_labels}")
    print(f"Total unique characters: {len(sorted_chars)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build OCR Vocabulary")
    parser.add_argument("--dataset-root", type=str, default="data/tamil_ocr_dataset")
    parser.add_argument("--language", type=str, default="tamil")
    args = parser.parse_args()
    
    build_vocabulary(args.dataset_root, args.language)
