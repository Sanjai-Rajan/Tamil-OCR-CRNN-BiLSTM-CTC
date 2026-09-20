import json
from pathlib import Path

class Tokenizer:
    def __init__(self, vocab_path=None, chars=None):
        """
        Args:
            vocab_path (str): Path to generated JSON vocabulary.
            chars (str): Fallback string of characters.
        """
        self.char_map = {}
        self.idx_map = {}
        self.num_classes = 0
        
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.char_map = vocab_data["char_to_index"]
            self.idx_map = {int(k): v for k, v in vocab_data["index_to_char"].items()}
            self.num_classes = len(self.char_map)
        elif chars:
            self.char_map = {c: i+1 for i, c in enumerate(chars)}
            self.char_map['<blank>'] = 0
            self.idx_map = {i: c for c, i in self.char_map.items()}
            self.num_classes = len(self.char_map)
        else:
            raise ValueError("Either vocab_path or chars must be provided.")

    def encode(self, text):
        """
        Encodes a UTF-8 string into a list of integers.
        """
        encoded = []
        for c in text:
            if c in self.char_map:
                encoded.append(self.char_map[c])
            else:
                raise ValueError(f"Unknown character '{c}' found in text: {text}")
        return encoded

    def decode(self, indices):
        """
        Decodes a list of integers back into a UTF-8 string.
        """
        return "".join([self.idx_map.get(idx, "") for idx in indices if idx != 0])

