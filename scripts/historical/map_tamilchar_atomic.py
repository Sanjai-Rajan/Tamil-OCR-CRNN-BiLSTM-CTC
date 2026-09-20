import json
import csv
from pathlib import Path
import unicodedata

def main():
    root = Path(__file__).resolve().parent.parent
    tamilchar_path = root / "data" / "TamilNet_old" / "TamilChar.csv"
    vocab_path = root / "data" / "tamil_ocr_dataset" / "vocabulary" / "tamil_vocab.json"
    mapping_path = root / "outputs" / "fast_track" / "ATOMIC_CHARACTER_MAPPING.csv"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab_chars = vocab_data.get("characters", [])
        
    vocab_set = set(vocab_chars)
    
    with open(tamilchar_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        
    header = lines[0].split(',')
    rows = lines[1:]
    
    results = []
    
    usable_classes = 0
    composite_classes = 0
    oov_classes = 0
    
    for row in rows:
        if not row: continue
        parts = row.split(',')
        label = parts[0].strip()
        img_name = parts[1].strip()
        
        # Determine status
        unicode_codepoints = " ".join([f"U+{ord(c):04X}" for c in label])
        
        # Is it in vocab?
        if label in vocab_set:
            status = "MATCH"
            atomic_char = label
            vocab_index = vocab_chars.index(label)
            reason = "Direct match in atomic vocabulary"
            usable_classes += 1
        else:
            # Check if all chars in label are in vocab
            all_in_vocab = all(c in vocab_set for c in label)
            if all_in_vocab and len(label) > 1:
                status = "COMPOSITE"
                atomic_char = ""
                vocab_index = -1
                reason = "Composed of valid atomic tokens, but represents multiple characters"
                composite_classes += 1
            else:
                status = "OOV"
                atomic_char = ""
                vocab_index = -1
                reason = "Contains out-of-vocabulary character(s)"
                oov_classes += 1
                
        results.append({
            "original_label": label,
            "unicode_codepoints": unicode_codepoints,
            "normalized_label": label,
            "atomic_character": atomic_char,
            "vocabulary_index": vocab_index,
            "status": status,
            "reason": reason
        })
        
    with open(mapping_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["original_label", "unicode_codepoints", "normalized_label", "atomic_character", "vocabulary_index", "status", "reason"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Total samples: {len(results)}")
    print(f"Unique labels: {len(set([r['original_label'] for r in results]))}")
    print(f"Atomic usable (MATCH): {usable_classes}")
    print(f"Composite (COMPOSITE): {composite_classes}")
    print(f"OOV (OOV): {oov_classes}")

if __name__ == "__main__":
    main()
