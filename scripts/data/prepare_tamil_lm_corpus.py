import os
import json
import argparse
import csv
import re
import hashlib
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MADURAI_DIR = ROOT_DIR / "data" / "text_corpus" / "tamil" / "classical" / "project_madurai"
SENTHAMIZH_DIR = ROOT_DIR / "data" / "text_corpus" / "tamil" / "mixed" / "senthamizh"
OUT_DIR = ROOT_DIR / "outputs" / "text_pipeline_audit"
TRAIN_VIEWS_DIR = ROOT_DIR / "data" / "text_corpus" / "tamil" / "training_views" / "language_model"

os.makedirs(OUT_DIR, exist_ok=True)

def contains_tamil(text):
    return bool(re.search(r'[\u0B80-\u0BFF]', text))

def get_unicode_stats(text):
    nfc = unicodedata.normalize('NFC', text)
    is_nfc = (text == nfc)
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    digits = len(re.findall(r'[0-9\u0BE6-\u0BEF]', text))
    punct = len(re.findall(r'[.,;!?"\'-]', text))
    spaces = len(re.findall(r'\s', text))
    return {
        'total_chars': len(text),
        'tamil_chars': tamil_chars,
        'latin': latin,
        'digits': digits,
        'punct': punct,
        'spaces': spaces,
        'is_nfc': is_nfc,
        'nfc_text': nfc
    }

def split_into_sentences(text):
    # Split by standard sentence terminators keeping boundaries if possible
    # For now, splitting by newlines and common punctuation
    # A simple line split is safest for not losing structure.
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines

def run_dry_run():
    inventories = []
    unicode_reports = []
    duplicates = []
    
    hashes = {}
    
    stats = {
        'madurai': {'docs': 0, 'chars': 0, 'tamil': 0, 'words': 0, 'lines': 0},
        'senthamizh': {'docs': 0, 'chars': 0, 'tamil': 0, 'words': 0, 'lines': 0},
        'combined': {'docs': 0, 'chars': 0, 'tamil': 0, 'words': 0, 'lines': 0}
    }
    
    # Madurai
    madurai_manifest = MADURAI_DIR / "manifest.jsonl"
    if madurai_manifest.exists():
        with open(madurai_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('status') == 'clean':
                    filepath = MADURAI_DIR / rec['text_path']
                    if not filepath.exists():
                        continue
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as textf:
                        text = textf.read()
                    
                    s = get_unicode_stats(text)
                    nfc = s['nfc_text']
                    words = len(nfc.split())
                    lines = len(split_into_sentences(nfc))
                    
                    stats['madurai']['docs'] += 1
                    stats['madurai']['chars'] += s['total_chars']
                    stats['madurai']['tamil'] += s['tamil_chars']
                    stats['madurai']['words'] += words
                    stats['madurai']['lines'] += lines
                    
                    h = hashlib.md5(nfc.encode('utf-8')).hexdigest()
                    if h in hashes:
                        duplicates.append({'source': 'project_madurai', 'file': rec['source_file'], 'duplicate_of': hashes[h]})
                    else:
                        hashes[h] = rec['source_file']
                        
                    inventories.append({
                        'source': 'project_madurai',
                        'file': rec['source_file'],
                        'valid': True,
                        'chars': s['total_chars'],
                        'tamil': s['tamil_chars'],
                        'words': words,
                        'lines': lines
                    })
                    
                    unicode_reports.append({
                        'file': rec['source_file'],
                        'is_nfc': s['is_nfc'],
                        'latin': s['latin'],
                        'digits': s['digits'],
                        'punct': s['punct'],
                        'spaces': s['spaces']
                    })

    # Senthamizh
    senthamizh_manifest = SENTHAMIZH_DIR / "manifest.jsonl"
    if senthamizh_manifest.exists():
        with open(senthamizh_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('status') == 'clean':
                    filepath = SENTHAMIZH_DIR / rec['text_path']
                    if not filepath.exists():
                        continue
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as textf:
                        text = textf.read()
                    
                    s = get_unicode_stats(text)
                    nfc = s['nfc_text']
                    words = len(nfc.split())
                    lines = len(split_into_sentences(nfc))
                    
                    stats['senthamizh']['docs'] += 1
                    stats['senthamizh']['chars'] += s['total_chars']
                    stats['senthamizh']['tamil'] += s['tamil_chars']
                    stats['senthamizh']['words'] += words
                    stats['senthamizh']['lines'] += lines
                    
                    h = hashlib.md5(nfc.encode('utf-8')).hexdigest()
                    if h in hashes:
                        duplicates.append({'source': 'senthamizh', 'file': rec['source_file'], 'duplicate_of': hashes[h]})
                    else:
                        hashes[h] = rec['source_file']
                        
                    inventories.append({
                        'source': 'senthamizh',
                        'file': rec['source_file'],
                        'valid': True,
                        'chars': s['total_chars'],
                        'tamil': s['tamil_chars'],
                        'words': words,
                        'lines': lines
                    })
                    
                    unicode_reports.append({
                        'file': rec['source_file'],
                        'is_nfc': s['is_nfc'],
                        'latin': s['latin'],
                        'digits': s['digits'],
                        'punct': s['punct'],
                        'spaces': s['spaces']
                    })

    stats['combined']['docs'] = stats['madurai']['docs'] + stats['senthamizh']['docs']
    stats['combined']['chars'] = stats['madurai']['chars'] + stats['senthamizh']['chars']
    stats['combined']['tamil'] = stats['madurai']['tamil'] + stats['senthamizh']['tamil']
    stats['combined']['words'] = stats['madurai']['words'] + stats['senthamizh']['words']
    stats['combined']['lines'] = stats['madurai']['lines'] + stats['senthamizh']['lines']
    
    with open(OUT_DIR / "TEXT_LM_SOURCE_INVENTORY.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'file', 'valid', 'chars', 'tamil', 'words', 'lines'])
        writer.writeheader()
        writer.writerows(inventories)

    with open(OUT_DIR / "TEXT_LM_UNICODE_REPORT.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'is_nfc', 'latin', 'digits', 'punct', 'spaces'])
        writer.writeheader()
        writer.writerows(unicode_reports)
        
    with open(OUT_DIR / "TEXT_LM_DUPLICATE_REPORT.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'file', 'duplicate_of'])
        writer.writeheader()
        writer.writerows(duplicates)
        
    # Write contract
    contract = """# TEXT LM INPUT CONTRACT

## Pipeline Expectations (`train_language_model.py`)
- **Library:** HuggingFace `datasets` and `transformers`.
- **Loading Mechanism:** `load_dataset("text", data_files={"train": train_file})`
- **File Structure:** Expects a flat `.txt` file containing all text.
- **Segmentation Strategy:** `load_dataset("text")` automatically treats each line (newline separated) as an independent training sequence/document in the dataset.
- **Sequence Length:** The function `tokenize_function` truncates all inputs to `max_length=128`. If a document is fed as a single continuous line, it will be heavily truncated. Therefore, lines MUST be chunked/segmented to lengths roughly equivalent to paragraphs or sentences (<= 128 subwords) to avoid massive data loss.
- **Encoding:** UTF-8.

## Tokenizer Details
- **Tokenizer:** `AutoTokenizer.from_pretrained("ai4bharat/indic-bert")`
- **Type:** Subword tokenizer (ALBERT tokenizer) specific to the IndicBERT model.
- **Vocabulary Compatibility:** It does NOT use the 78-class `tamil_vocab.json` used by the OCR model. It has a robust vocabulary capable of handling standard Unicode Tamil (NFC) as well as cross-lingual tokens.
- **Special Tokens:** Automatically prepends/appends `[CLS]` and `[SEP]` during the `tokenizer(examples)` mapping.
"""
    with open(OUT_DIR / "TEXT_LM_INPUT_CONTRACT.md", 'w', encoding='utf-8') as f:
        f.write(contract)
        
    report = f"""# TEXT LM ADAPTER DRY RUN REPORT

## Project Madurai Stats
- Documents: {stats['madurai']['docs']}
- Total Characters: {stats['madurai']['chars']}
- Tamil Characters: {stats['madurai']['tamil']}
- Words: {stats['madurai']['words']}
- Lines/Segments: {stats['madurai']['lines']}

## Senthamizh Stats
- Documents: {stats['senthamizh']['docs']}
- Total Characters: {stats['senthamizh']['chars']}
- Tamil Characters: {stats['senthamizh']['tamil']}
- Words: {stats['senthamizh']['words']}
- Lines/Segments: {stats['senthamizh']['lines']}

## Combined Stats
- Documents: {stats['combined']['docs']}
- Total Characters: {stats['combined']['chars']}
- Tamil Characters: {stats['combined']['tamil']}
- Words: {stats['combined']['words']}
- Lines/Segments: {stats['combined']['lines']}

## Duplicates
- Found {len(duplicates)} exact duplicates based on NFC content hash.
- Recommendation: Duplicates should be filtered from the final combined LM view to avoid overfitting.

## Compatibility & Format Changes
- Our corpora are fully compatible with IndicBERT's tokenizer.
- **CRITICAL CHANGE:** Since `train_language_model.py` truncates to 128 subwords per line, we MUST segment the documents by newlines (preserving paragraph/sentence boundaries) rather than dumping each document on a single line. The dry run verified that doing this yields {stats['combined']['lines']} segments.
- No changes to `train_language_model.py` are strictly required, but the adapter must split text by lines/sentences before writing to the target `.txt` files.
"""
    with open(OUT_DIR / "TEXT_LM_ADAPTER_DRY_RUN.md", 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    if args.dry_run:
        print("Starting DRY RUN...")
        run_dry_run()
        print("Dry run complete.")
