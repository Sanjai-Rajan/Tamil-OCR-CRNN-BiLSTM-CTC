import json
import unicodedata
import os
from pathlib import Path

class DeterministicCorrector:
    def __init__(self, vocab_path: str = None):
        if vocab_path is None:
            # Default to the generated vocabulary
            root_dir = Path(__file__).resolve().parent.parent.parent
            vocab_path = root_dir / 'data' / 'text_corpus' / 'tamil_correction_vocabulary.json'
            
        self.vocab = {}
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        
        # Valid Tamil character list for edit operations
        self.tamil_chars = [chr(c) for c in range(0x0B80, 0x0C00)]
        
    def _normalize(self, word: str) -> str:
        # Normalize and remove non-Tamil characters if necessary
        return unicodedata.normalize('NFKC', word)
        
    def _edits1(self, word):
        """Generate candidates that are one edit distance away from the word."""
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in self.tamil_chars]
        inserts    = [L + c + R               for L, R in splits for c in self.tamil_chars]
        return set(deletes + transposes + replaces + inserts)
        
    def correct(self, ocr_word: str) -> tuple[str, str]:
        """
        Corrects an OCR word using a deterministic vocabulary.
        Returns:
            (corrected_word, status)
            status can be 'unchanged_vocab_match', 'corrected', or 'unchanged_no_safe_correction'
        """
        if not self.vocab:
            # If no vocab is loaded, fall back safely
            return ocr_word, 'unchanged_no_vocab'
            
        norm_word = self._normalize(ocr_word)
        
        # 1. Exact Match
        if norm_word in self.vocab:
            return norm_word, 'unchanged_vocab_match'
            
        # 2. Candidate Generation (Edit Distance 1)
        # To avoid unbounded fuzzy matching, we only look for candidates 1 edit away
        candidates = self._edits1(norm_word)
        
        # 3. Candidate Selection
        valid_candidates = []
        for cand in candidates:
            if cand in self.vocab:
                valid_candidates.append(cand)
                
        if not valid_candidates:
            return ocr_word, 'unchanged_no_safe_correction'
            
        # Sort candidates by frequency in descending order
        valid_candidates.sort(key=lambda w: self.vocab[w], reverse=True)
        best_candidate = valid_candidates[0]
        
        # Require a minimum evidence frequency to accept the correction
        # Let's say the word must appear at least 10 times in the corpus to be a "safe" correction
        min_evidence_threshold = 10
        if self.vocab[best_candidate] >= min_evidence_threshold:
            return best_candidate, 'corrected'
            
        # Fallback if the candidate is too rare to be reliable
        return ocr_word, 'unchanged_no_safe_correction'
