import os
import json
import csv
from pathlib import Path

class AnnotationPipeline:
    def __init__(self, dataset_root="dataset"):
        self.dataset_root = Path(dataset_root)
        self.annotations_dir = self.dataset_root / "annotations"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
    def load_csv_annotations(self, csv_path):
        """
        Loads annotations from a CSV file (e.g., filename, transcription)
        """
        labels_dict = {}
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                if len(row) >= 2:
                    filename, text = row[0], row[1]
                    labels_dict[filename] = text
        return labels_dict
        
    def load_json_annotations(self, json_path):
        """
        Loads annotations from a JSON file format.
        """
        with open(json_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def merge_datasets(self, arabic_dict, tamil_dict):
        """
        Merges multi-lingual dictionaries into a unified training dict.
        """
        merged = {}
        merged.update(arabic_dict)
        merged.update(tamil_dict)
        return merged
        
    def generate_character_set(self, labels_dict):
        """
        Scans all text annotations to generate a unique UTF-8 character set
        for tokenizer initialization.
        """
        unique_chars = set()
        for text in labels_dict.values():
            unique_chars.update(list(text))
        return "".join(sorted(list(unique_chars)))

if __name__ == "__main__":
    # Example usage
    pipeline = AnnotationPipeline()
    print("Annotation pipeline initialized.")
