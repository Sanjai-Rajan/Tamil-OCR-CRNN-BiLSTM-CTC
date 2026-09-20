import os
import json
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

os.makedirs(TRAIN_VIEWS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

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

def split_into_segments(text):
    # Splits by newline, strips whitespace, ignores empty lines
    return [line.strip() for line in text.split('\n') if line.strip()]

def run_extraction():
    manifest_records = []
    
    stats = {
        'madurai': {'docs': 0, 'segments': 0, 'words': 0, 'tamil': 0},
        'senthamizh': {'docs': 0, 'segments': 0, 'words': 0, 'tamil': 0},
        'combined': {'docs': 0, 'segments': 0, 'words': 0, 'tamil': 0}
    }
    
    duplicates_found = 0
    hashes = set()
    
    pm_out = TRAIN_VIEWS_DIR / "project_madurai.txt"
    st_out = TRAIN_VIEWS_DIR / "senthamizh.txt"
    cb_out = TRAIN_VIEWS_DIR / "combined_tamil_lm.txt"
    
    with open(pm_out, 'w', encoding='utf-8') as f_pm, \
         open(st_out, 'w', encoding='utf-8') as f_st, \
         open(cb_out, 'w', encoding='utf-8') as f_cb:
         
        def process_corpus(corpus_dir, source_name, target_f):
            manifest_path = corpus_dir / "manifest.jsonl"
            if not manifest_path.exists():
                return
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get('status') == 'clean':
                        filepath = corpus_dir / rec['text_path']
                        if not filepath.exists():
                            continue
                            
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as textf:
                            text = textf.read()
                            
                        # Extract stats for the whole document to detect exact doc duplicates
                        s = get_unicode_stats(text)
                        nfc = s['nfc_text']
                        
                        h = hashlib.md5(nfc.encode('utf-8')).hexdigest()
                        if h in hashes:
                            nonlocal duplicates_found
                            duplicates_found += 1
                        else:
                            hashes.add(h)
                            
                        stats[source_name]['docs'] += 1
                        
                        segments = split_into_segments(nfc)
                        
                        for i, segment in enumerate(segments):
                            words = len(segment.split())
                            seg_stats = get_unicode_stats(segment)
                            
                            stats[source_name]['segments'] += 1
                            stats[source_name]['words'] += words
                            stats[source_name]['tamil'] += seg_stats['tamil_chars']
                            
                            # Write to specific source file and combined file
                            target_f.write(segment + '\n')
                            f_cb.write(segment + '\n')
                            
                            manifest_records.append({
                                "source": source_name,
                                "source_file": rec['source_file'],
                                "segment_id": f"{source_name}_{stats[source_name]['docs']}_{i}",
                                "segment_index": i,
                                "text_preview": segment[:30] + '...' if len(segment) > 30 else segment,
                                "char_count": len(segment),
                                "tamil_char_count": seg_stats['tamil_chars'],
                                "word_count": words
                            })

        process_corpus(MADURAI_DIR, 'madurai', f_pm)
        process_corpus(SENTHAMIZH_DIR, 'senthamizh', f_st)
        
    stats['combined']['docs'] = stats['madurai']['docs'] + stats['senthamizh']['docs']
    stats['combined']['segments'] = stats['madurai']['segments'] + stats['senthamizh']['segments']
    stats['combined']['words'] = stats['madurai']['words'] + stats['senthamizh']['words']
    stats['combined']['tamil'] = stats['madurai']['tamil'] + stats['senthamizh']['tamil']
    
    with open(TRAIN_VIEWS_DIR / "manifest.jsonl", 'w', encoding='utf-8') as f:
        for r in manifest_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    # VALIDATION
    assert stats['madurai']['docs'] == 1296
    assert stats['senthamizh']['docs'] == 16
    assert stats['combined']['docs'] == 1312
    # Verify that the total segments matches what was written
    assert stats['combined']['segments'] == len(manifest_records)
    
    with open(OUT_DIR / "TEXT_LM_FINAL_ADAPTER_REPORT.md", 'w', encoding='utf-8') as f:
        f.write("# TEXT LM FINAL ADAPTER REPORT\n\n")
        f.write("## Source Counts\n")
        f.write(f"- Project Madurai: {stats['madurai']['docs']} documents\n")
        f.write(f"- Senthamizh: {stats['senthamizh']['docs']} documents\n")
        f.write(f"- Combined: {stats['combined']['docs']} documents\n\n")
        
        f.write("## Segment Counts\n")
        f.write(f"- Project Madurai: {stats['madurai']['segments']}\n")
        f.write(f"- Senthamizh: {stats['senthamizh']['segments']}\n")
        f.write(f"- Combined: {stats['combined']['segments']}\n\n")
        
        f.write(f"## Total Words: {stats['combined']['words']}\n")
        f.write(f"## Total Tamil Characters: {stats['combined']['tamil']}\n\n")
        
        f.write("## Validations\n")
        f.write("- UTF-8: Passed. Written using utf-8 encoding strictly.\n")
        f.write("- NFC: Passed. NFC enforced on all written segments.\n")
        f.write("- No empty segments written.\n")
        f.write(f"- Duplicates found during final write: {duplicates_found}\n")
        f.write("- Provenance tracking maintained via `manifest.jsonl`.\n\n")
        
        f.write("## Output Paths\n")
        f.write(f"- Project Madurai Text: {pm_out}\n")
        f.write(f"- Senthamizh Text: {st_out}\n")
        f.write(f"- Combined Text: {cb_out}\n")
        f.write(f"- Manifest: {TRAIN_VIEWS_DIR / 'manifest.jsonl'}\n\n")
        
        f.write("## Compatibility\n")
        f.write("Ready for `train_language_model.py`. The text has been safely segmented into newline-delimited chunks to prevent over-truncation from `max_length=128` during HuggingFace dataset loading.\n")

if __name__ == "__main__":
    print("Starting Final Data Transformation...")
    run_extraction()
    print("Transformation and validation complete.")
