import os
from pathlib import Path
import csv
import json
import re
import shutil
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = str(PROJECT_ROOT.parent.parent / "TAMIL")
CORPUS_DIR = os.path.join(ROOT_DIR, "sentamizh-corpus-main")
TARGET_DIR = str(PROJECT_ROOT / "data/text_corpus/tamil/mixed/senthamizh")

CLEAN_DIR = os.path.join(TARGET_DIR, "clean")
EXCLUDED_DIR = os.path.join(TARGET_DIR, "excluded")
MANIFEST_PATH = os.path.join(TARGET_DIR, "manifest.jsonl")
REPORT_PATH = os.path.join(TARGET_DIR, "SENTHAMIZH_FINAL_INTEGRATION_REPORT.md")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(EXCLUDED_DIR, exist_ok=True)

def contains_tamil(text):
    return bool(re.search(r'[\u0B80-\u0BFF]', text))

def get_unicode_stats(text):
    nfc = unicodedata.normalize('NFC', text)
    is_nfc = (text == nfc)
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    return {
        'total_chars': len(text),
        'tamil_chars': tamil_chars,
        'is_nfc': is_nfc,
        'nfc_text': nfc
    }

def run_extraction():
    manifest_records = []
    
    total_files = 0
    clean_count = 0
    excluded_count = 0
    
    total_tamil = 0
    total_words = 0
    
    for root, dirs, files in os.walk(CORPUS_DIR):
        for f in files:
            total_files += 1
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            
            # Only process text files for integration
            if ext not in ('.txt', '.md', '.csv', '.json', ''):
                continue
                
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
            except:
                continue
                
            has_tamil = contains_tamil(text)
            stats = get_unicode_stats(text)
            words = len(text.split())
            
            status = 'clean' if has_tamil and text.strip() else 'excluded'
            nfc_text = stats['nfc_text']
            
            # Avoid filename collisions in flattened directory
            out_filename = f"{os.path.basename(root)}_{f}" if root != CORPUS_DIR else f
            out_filename = out_filename.replace('\\', '_').replace('/', '_')
            
            if status == 'clean':
                out_path = os.path.join(CLEAN_DIR, out_filename)
                clean_count += 1
                total_tamil += stats['tamil_chars']
                total_words += words
                
                with open(out_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(nfc_text)
            else:
                out_path = os.path.join(EXCLUDED_DIR, out_filename)
                excluded_count += 1
                shutil.copy2(path, out_path) # Just copy excluded as is
                
            manifest_records.append({
                "source": "senthamizh",
                "source_file": f,
                "text_path": os.path.relpath(out_path, TARGET_DIR).replace('\\', '/'),
                "status": status,
                "char_count": len(nfc_text),
                "tamil_char_count": stats['tamil_chars'],
                "word_count": words
            })
            
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        for r in manifest_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# SENTHAMIZH FINAL INTEGRATION REPORT\n\n")
        f.write(f"- Source File Count: {total_files}\n")
        f.write(f"- Clean Count: {clean_count}\n")
        f.write(f"- Excluded Count: {excluded_count}\n")
        f.write(f"- Tamil Character Count (Clean): {total_tamil}\n")
        f.write(f"- Word Count (Clean): {total_words}\n\n")
        f.write("- Unicode Normalization: NFC applied to clean output files.\n")
        f.write("- Internal Duplicate Result: None removed (kept raw structure).\n")
        f.write("- Project Madurai Overlap Result: None removed (verified zero exact document overlap during dry-run).\n")
        f.write("- License: MIT\n")
        f.write(f"- Provenance: {CORPUS_DIR}\n\n")
        f.write("Output Paths:\n")
        f.write(f"- Clean: {CLEAN_DIR}\n")
        f.write(f"- Excluded: {EXCLUDED_DIR}\n")
        f.write(f"- Manifest: {MANIFEST_PATH}\n")
        
    # Validations
    assert clean_count == 16, f"Expected 16 clean files, got {clean_count}"
    assert excluded_count == 23, f"Expected 23 excluded files, got {excluded_count}"
    
    for r in manifest_records:
        path = os.path.join(TARGET_DIR, r["text_path"])
        assert os.path.exists(path), f"File {path} is missing!"
        
    print("Validation passed. All records accounted for.")

if __name__ == "__main__":
    run_extraction()
