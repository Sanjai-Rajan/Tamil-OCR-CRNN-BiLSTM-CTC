import os
import re
import csv
import hashlib
import unicodedata
from collections import defaultdict
from html.parser import HTMLParser

RAW_DIR = str(PROJECT_ROOT.parent.parent / "TAMIL/Project_Madurai_Tamil_Corpus/raw_html")
OUTPUT_DIR = str(PROJECT_ROOT.parent.parent / "TAMIL/deep_audit/project_madurai_dry_run")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.in_script = False
        
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'head', 'title', 'meta', 'link'):
            self.in_script = True
        elif tag in ('p', 'br', 'div', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th'):
            self.text_parts.append('\n')
            
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head', 'title', 'meta', 'link'):
            self.in_script = False
        elif tag in ('p', 'br', 'div', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th'):
            self.text_parts.append('\n')
            
    def handle_data(self, data):
        if not self.in_script:
            self.text_parts.append(data)
            
    def get_text(self):
        text = ''.join(self.text_parts)
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join([line for line in lines if line])

def extract_text_from_html(html_content):
    parser = TextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text()
    except:
        return re.sub(r'<[^>]+>', '\n', html_content)

def contains_tamil(text):
    return bool(re.search(r'[\u0B80-\u0BFF]', text))
    
def get_unicode_stats(text):
    nfc = unicodedata.normalize('NFC', text)
    is_nfc = (text == nfc)
    
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    combining_marks = len(re.findall(r'[\u0BBE-\u0BCC\u0BD7]', text))
    virama = len(re.findall(r'\u0BCD', text))
    zero_width = len(re.findall(r'[\u200B-\u200D\uFEFF]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    digits = len(re.findall(r'[0-9\u0BE6-\u0BEF]', text))
    punct = len(re.findall(r'[.,;!?"\'-]', text))
    
    return {
        'total_chars': len(text),
        'tamil_chars': tamil_chars,
        'combining': combining_marks,
        'virama': virama,
        'zero_width': zero_width,
        'latin': latin,
        'digits': digits,
        'punct': punct,
        'is_nfc': is_nfc
    }

def analyze_corpus():
    inventory = []
    unicode_report = []
    line_counts = defaultdict(int)
    hash_map = defaultdict(list)
    
    total_files = 0
    parsed = 0
    tamil_bearing = 0
    empty = 0
    meta_only = 0
    review_needed = 0
    failed = 0
    
    total_chars = 0
    total_tamil = 0
    total_words = 0
    
    raw_size = 0
    clean_size = 0
    
    sample_count = 0
    
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if not f.endswith('.html'):
                continue
                
            total_files += 1
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            raw_size += size
            
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    html_content = file.read()
            except Exception as e:
                failed += 1
                inventory.append({'file': f, 'class': 'FAILED_EXTRACTION', 'warnings': str(e)})
                continue
                
            is_html = '<html' in html_content.lower() or '<body' in html_content.lower() or '<head' in html_content.lower()
            
            if is_html:
                text = extract_text_from_html(html_content)
            else:
                text = html_content
                
            clean_size += len(text.encode('utf-8'))
            
            for line in set(text.split('\n')):
                if 10 < len(line) < 200:
                    line_counts[line] += 1
            
            h = hashlib.md5(text.encode('utf-8')).hexdigest()
            hash_map[h].append(f)
            
            parsed += 1
            
            if not text.strip():
                empty += 1
                inventory.append({'file': f, 'class': 'EMPTY', 'warnings': ''})
                continue
                
            has_tamil = contains_tamil(text)
            stats = get_unicode_stats(text)
            words = len(text.split())
            
            total_chars += stats['total_chars']
            total_tamil += stats['tamil_chars']
            total_words += words
            
            classification = 'VALID_TAMIL_TEXT'
            warning = ''
            
            if not has_tamil:
                classification = 'METADATA_ONLY'
                meta_only += 1
            else:
                tamil_bearing += 1
                if stats['tamil_chars'] < 100:
                    classification = 'PARTIAL_TAMIL_TEXT'
                    review_needed += 1
                elif stats['latin'] > stats['tamil_chars']:
                    classification = 'MANUAL_REVIEW'
                    warning = 'High Latin ratio'
                    review_needed += 1
                    
            if not stats['is_nfc']:
                warning += ' Needs NFC Normalization.'
                
            inventory.append({
                'file': f,
                'class': classification,
                'extracted_chars': stats['total_chars'],
                'tamil_chars': stats['tamil_chars'],
                'words': words,
                'warnings': warning.strip()
            })
            
            unicode_report.append({
                'file': f,
                **stats
            })
            
            if classification == 'VALID_TAMIL_TEXT' and sample_count < 10:
                with open(os.path.join(SAMPLES_DIR, f"{f}.txt"), 'w', encoding='utf-8') as out_f:
                    out_f.write(text)
                sample_count += 1
                
    boilerplate = [{'line': k, 'count': v} for k, v in line_counts.items() if v > 10]
    boilerplate.sort(key=lambda x: x['count'], reverse=True)
    
    duplicates = [{'hash': k, 'files': ' | '.join(v)} for k, v in hash_map.items() if len(v) > 1]
    
    with open(os.path.join(OUTPUT_DIR, "PM_EXTRACTION_INVENTORY.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'class', 'extracted_chars', 'tamil_chars', 'words', 'warnings'])
        writer.writeheader()
        writer.writerows(inventory)
        
    with open(os.path.join(OUTPUT_DIR, "PM_BOILERPLATE_CANDIDATES.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['count', 'line'])
        writer.writeheader()
        for b in boilerplate:
            writer.writerow(b)
            
    with open(os.path.join(OUTPUT_DIR, "PM_DUPLICATES.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['hash', 'files'])
        writer.writeheader()
        for d in duplicates:
            writer.writerow(d)
            
    with open(os.path.join(OUTPUT_DIR, "PM_UNICODE_REPORT.csv"), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'total_chars', 'tamil_chars', 'combining', 'virama', 'zero_width', 'latin', 'digits', 'punct', 'is_nfc'])
        writer.writeheader()
        writer.writerows(unicode_report)
        
    with open(os.path.join(OUTPUT_DIR, "PM_DRY_RUN_REPORT.md"), 'w', encoding='utf-8') as f:
        f.write("# Project Madurai Preprocessing Dry Run Report\n\n")
        f.write("## Overview\n")
        f.write(f"- input file count: {total_files}\n")
        f.write(f"- successfully parsed: {parsed}\n")
        f.write(f"- Tamil-bearing: {tamil_bearing}\n")
        f.write(f"- empty: {empty}\n")
        f.write(f"- metadata-only: {meta_only}\n")
        f.write(f"- manual-review: {review_needed}\n")
        f.write(f"- failed extraction: {failed}\n\n")
        
        f.write("## Corpus Stats\n")
        f.write(f"- total extracted characters: {total_chars}\n")
        f.write(f"- total Tamil characters: {total_tamil}\n")
        f.write(f"- total words: {total_words}\n")
        f.write(f"- duplicate count: {len(duplicates)}\n")
        f.write(f"- estimated boilerplate amount: {len(boilerplate)} common lines detected\n\n")
        
        f.write("## Before / After\n")
        f.write(f"- BEFORE:\n  raw HTML size/files: {total_files} files, {raw_size/1024/1024:.2f} MB\n")
        f.write(f"- AFTER:\n  estimated clean-text files/size/characters/words: {tamil_bearing} files, {clean_size/1024/1024:.2f} MB, {total_chars} chars, {total_tamil} Tamil chars, {total_words} words\n\n")
        
        f.write("## Unicode Findings\n")
        f.write("Checked for NFC normalization requirements. See PM_UNICODE_REPORT.csv for details.\n")
        
if __name__ == "__main__":
    analyze_corpus()
