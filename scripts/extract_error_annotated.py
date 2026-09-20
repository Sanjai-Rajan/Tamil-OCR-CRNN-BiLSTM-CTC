import os
from pathlib import Path
import csv
import json
import collections
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = str(PROJECT_ROOT.parent.parent / "TAMIL")
CORPUS_DIR = os.path.join(ROOT_DIR, "Error Annotated Tamil Corpus")
CSV_PATH = os.path.join(CORPUS_DIR, "Error Annotated Corpus.csv")
TARGET_DIR = str(PROJECT_ROOT / "data/text_corpus/tamil/restoration/error_annotated")

PAIRS_PATH = os.path.join(TARGET_DIR, "pairs.jsonl")
CONFLICTS_PATH = os.path.join(TARGET_DIR, "review_conflicts.jsonl")
EXCLUDED_PATH = os.path.join(TARGET_DIR, "excluded_unpaired.jsonl")
METADATA_PATH = os.path.join(TARGET_DIR, "metadata.json")
REPORT_PATH = os.path.join(TARGET_DIR, "ERROR_ANNOTATED_FINAL_INTEGRATION_REPORT.md")

os.makedirs(TARGET_DIR, exist_ok=True)

def run_extraction():
    records = []
    
    noisy_to_clean = collections.defaultdict(set)
    clean_to_noisy = collections.defaultdict(set)
    
    with open(CSV_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        row_id = 1
        for row in reader:
            if not row or len(row) < 3:
                row_id += 1
                continue
                
            noisy = row[0].strip()
            clean = row[1].strip()
            cat = row[2].strip()
            
            noisy_nfc = unicodedata.normalize('NFC', noisy) if noisy else ""
            clean_nfc = unicodedata.normalize('NFC', clean) if clean else ""
            cat_nfc = unicodedata.normalize('NFC', cat) if cat else ""
            
            records.append({
                'id': row_id,
                'noisy': noisy_nfc,
                'clean': clean_nfc,
                'category': cat_nfc
            })
            
            if noisy_nfc and clean_nfc:
                noisy_to_clean[noisy_nfc].add(clean_nfc)
                clean_to_noisy[clean_nfc].add(noisy_nfc)
                
            row_id += 1

    clean_pairs = []
    conflict_records = []
    excluded_records = []
    
    seen_exact_pairs = set()
    dup_removed = 0
    
    cat_dist = collections.defaultdict(int)

    for r in records:
        n = r['noisy']
        c = r['clean']
        
        if n and not c:
            excluded_records.append(r)
        elif c and not n:
            excluded_records.append(r)
        elif n and c:
            # Check for conflict
            is_conflict = len(noisy_to_clean[n]) > 1 or len(clean_to_noisy[c]) > 1
            if is_conflict:
                conflict_records.append(r)
            else:
                pair_tuple = (n, c, r['category'])
                if pair_tuple in seen_exact_pairs:
                    dup_removed += 1
                else:
                    seen_exact_pairs.add(pair_tuple)
                    clean_pairs.append(r)
                    cat_dist[r['category']] += 1

    # Write outputs
    with open(PAIRS_PATH, 'w', encoding='utf-8') as f:
        for r in clean_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    with open(CONFLICTS_PATH, 'w', encoding='utf-8') as f:
        for r in conflict_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    with open(EXCLUDED_PATH, 'w', encoding='utf-8') as f:
        for r in excluded_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    # Metadata
    metadata = {
        "dataset_name": "Error Annotated Tamil Corpus",
        "description": "Linguistic error correction dataset",
        "license": "UNKNOWN",
        "provenance": "C:\\Users\\prsan\\Desktop\\TAMIL\\Error Annotated Tamil Corpus",
        "clean_pairs_count": len(clean_pairs),
        "categories": list(cat_dist.keys())
    }
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    # Report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# ERROR ANNOTATED FINAL INTEGRATION REPORT\n\n")
        f.write("**IMPORTANT NOTE**: This is a LINGUISTIC/CONTEXTUAL RESTORATION dataset, NOT a visual OCR-error dataset. The annotations represent grammatical and spelling corrections.\n\n")
        f.write(f"- Source Total Records: {len(records)}\n")
        f.write(f"- Final Clean Pairs (pairs.jsonl): {len(clean_pairs)}\n")
        f.write(f"- Duplicate Pairs Removed: {dup_removed}\n")
        f.write(f"- Conflict Records (review_conflicts.jsonl): {len(conflict_records)}\n")
        f.write(f"- Unpaired Records (excluded_unpaired.jsonl): {len(excluded_records)}\n\n")
        f.write("- Unicode Normalization: NFC normalization applied to derived output only.\n")
        f.write("- Provenance: C:\\Users\\prsan\\Desktop\\TAMIL\\Error Annotated Tamil Corpus\n")
        f.write("- License Status: UNKNOWN\n\n")
        f.write("## Category Distribution (Clean Pairs)\n")
        for cat, count in sorted(cat_dist.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {cat}: {count}\n")
            
if __name__ == "__main__":
    run_extraction()
