import random
from typing import List, Dict, Any

class DatasetSplitter:
    """
    Splits datasets deterministically into train, validation, and test sets.
    Ensures no data leakage by splitting at the document/page level when possible.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(self.seed)

    def split_manifest(self, manifest_items: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Splits a flat list of manifest items.
        For OCR, this should group by 'document_id' first to prevent leakage.
        """
        pass

if __name__ == "__main__":
    splitter = DatasetSplitter()
    print("Splitter interface ready.")
