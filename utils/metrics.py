import Levenshtein

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculates Character Error Rate."""
    if len(reference) == 0:
        if len(hypothesis) == 0:
            raise ValueError("Both reference and hypothesis are empty. Tokenizer failure likely.")
        return 1.0
    return Levenshtein.distance(reference, hypothesis) / len(reference)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            raise ValueError("Both reference and hypothesis are empty. Tokenizer failure likely.")
        return 1.0
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)
