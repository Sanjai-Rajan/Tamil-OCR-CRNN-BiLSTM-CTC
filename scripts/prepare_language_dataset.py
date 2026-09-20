import os
import random
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.language_understanding.tamil_preprocessor import TamilPreprocessor

def prepare_language_dataset(raw_dir: str, output_dir: str, val_ratio=0.1, test_ratio=0.1):
    """
    Reads raw Tamil text files, cleans them using TamilPreprocessor, 
    splits them into sentences, and saves them to train/val/test files.
    """
    preprocessor = TamilPreprocessor()
    
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
    if not raw_files:
        print(f"No .txt files found in {raw_dir}")
        return

    all_sentences = []
    print("Processing raw corpus files...")
    for filename in tqdm(raw_files):
        with open(os.path.join(raw_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            cleaned_text = preprocessor.preprocess(text)
            sentences = preprocessor.segment_sentences(cleaned_text)
            all_sentences.extend(sentences)
            
    print(f"Total sentences extracted: {len(all_sentences)}")
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(all_sentences)
    
    num_test = int(len(all_sentences) * test_ratio)
    num_val = int(len(all_sentences) * val_ratio)
    num_train = len(all_sentences) - num_test - num_val
    
    train_sentences = all_sentences[:num_train]
    val_sentences = all_sentences[num_train:num_train+num_val]
    test_sentences = all_sentences[num_train+num_val:]
    
    def save_split(sentences, split_name):
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        out_path = os.path.join(split_dir, f"{split_name}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            for s in sentences:
                f.write(s + "\n")
        print(f"Saved {len(sentences)} sentences to {out_path}")

    save_split(train_sentences, 'train')
    save_split(val_sentences, 'validation')
    save_split(test_sentences, 'test')
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/tamil_corpus/raw", help="Directory with raw text files")
    parser.add_argument("--output_dir", default="data/tamil_corpus", help="Output base directory")
    args = parser.parse_args()
    
    prepare_language_dataset(args.raw_dir, args.output_dir)
