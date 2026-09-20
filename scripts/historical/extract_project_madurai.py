import os
from pathlib import Path
import re
import json
import unicodedata
import urllib.parse
from html.parser import HTMLParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = str(PROJECT_ROOT.parent.parent / "TAMIL/Project_Madurai_Tamil_Corpus/raw_html")
TARGET_DIR = str(PROJECT_ROOT / "data/text_corpus/tamil/classical/project_madurai")
CLEAN_DIR = os.path.join(TARGET_DIR, "clean")
REVIEW_DIR = os.path.join(TARGET_DIR, "manual_review")
EXCLUDE_DIR = os.path.join(TARGET_DIR, "excluded")
MANIFEST_PATH = os.path.join(TARGET_DIR, "manifest.jsonl")
REPORT_PATH = os.path.join(TARGET_DIR, "PROJECT_MADURAI_FINAL_INTEGRATION_REPORT.md")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)
os.makedirs(EXCLUDE_DIR, exist_ok=True)

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
    latin = len(re.findall(r'[a-zA-Z]', text))
    
    return {
        'total_chars': len(text),
        'tamil_chars': tamil_chars,
        'latin': latin,
        'is_nfc': is_nfc,
        'nfc_text': nfc
    }

def remove_boilerplate(text):
    # Safe boilerplate removal for common Project Madurai headers/footers
    boilerplate_lines = [
        "Project Madurai is an open",
        "Preparation of HTML and PDF versions",
        "This page was last updated on",
        "Web version by",
        "Etext Preparation",
        "Proof-reading:",
        "(c) Project Madurai"
    ]
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        if not any(bp.lower() in line.lower() for bp in boilerplate_lines):
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def run_extraction():
    manifest_records = []
    
    total_files = 0
    clean_count = 0
    review_count = 0
    exclude_count = 0
    failed_count = 0
    
    total_tamil = 0
    total_words = 0
    
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if not f.endswith('.html'):
                continue
                
            total_files += 1
            path = os.path.join(root, f)
            
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    html_content = file.read()
            except Exception as e:
                failed_count += 1
                continue
                
            is_html = '<html' in html_content.lower() or '<body' in html_content.lower() or '<head' in html_content.lower()
            
            if is_html:
                text = extract_text_from_html(html_content)
            else:
                text = html_content
                
            text = remove_boilerplate(text)
            
            has_tamil = contains_tamil(text)
            stats = get_unicode_stats(text)
            words = len(text.split())
            nfc_text = stats['nfc_text']
            
            status = 'clean'
            if not has_tamil:
                status = 'excluded'
            elif not text.strip():
                status = 'excluded'
            elif stats['tamil_chars'] < 100 or stats['latin'] > stats['tamil_chars']:
                status = 'manual_review'
                
            out_filename = f.replace('.html', '.txt')
            if status == 'clean':
                out_path = os.path.join(CLEAN_DIR, out_filename)
                clean_count += 1
                total_tamil += stats['tamil_chars']
                total_words += words
            elif status == 'manual_review':
                out_path = os.path.join(REVIEW_DIR, out_filename)
                review_count += 1
            else:
                out_path = os.path.join(EXCLUDE_DIR, out_filename)
                exclude_count += 1
                
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write(nfc_text)
                
            url = f"https://www.projectmadurai.org/pm_etexts/utf8/{f}"
                
            manifest_records.append({
                "source": "project_madurai",
                "source_file": f,
                "source_url": url,
                "status": status,
                "text_path": os.path.relpath(out_path, TARGET_DIR).replace('\\', '/'),
                "char_count": len(nfc_text),
                "tamil_char_count": stats['tamil_chars'],
                "word_count": words
            })
            
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        for r in manifest_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# PROJECT MADURAI FINAL INTEGRATION REPORT\n\n")
        f.write(f"- Input Files: {total_files}\n")
        f.write(f"- Clean Files: {clean_count}\n")
        f.write(f"- Manual Review Files: {review_count}\n")
        f.write(f"- Excluded Files: {exclude_count}\n")
        f.write(f"- Failed Extractions: {failed_count}\n\n")
        f.write(f"- Total Tamil Characters (Clean): {total_tamil}\n")
        f.write(f"- Total Words (Clean): {total_words}\n\n")
        f.write("- Unicode Normalization: All output texts have been converted to NFC. Raw HTML remains unmodified.\n")
        f.write("- Boilerplate Removal: Removed common Project Madurai headers and footers.\n")
        f.write("- Duplicate Handling: None (kept raw structure).\n")
        f.write("- Source Provenance: Preserved in manifest.jsonl.\n\n")
        f.write("Output Paths:\n")
        f.write(f"- Clean: {CLEAN_DIR}\n")
        f.write(f"- Manual Review: {REVIEW_DIR}\n")
        f.write(f"- Excluded: {EXCLUDE_DIR}\n")
        f.write(f"- Manifest: {MANIFEST_PATH}\n")
        
    # Validation checks
    assert len(manifest_records) == total_files, "Manifest count mismatch"
    for r in manifest_records:
        path = os.path.join(TARGET_DIR, r["text_path"])
        assert os.path.exists(path), f"File {path} is missing!"
        
    print("Validation passed. All records accounted for.")

if __name__ == "__main__":
    run_extraction()
