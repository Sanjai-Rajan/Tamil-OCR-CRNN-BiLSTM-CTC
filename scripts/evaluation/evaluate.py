from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Using python-Levenshtein or jiwer for CER/WER is recommended, but we implement basic metric structure
import Levenshtein

def character_error_rate(reference, hypothesis):
    """
    Computes Character Error Rate (CER).
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    return Levenshtein.distance(reference, hypothesis) / len(reference)

def word_error_rate(reference, hypothesis):
    """
    Computes Word Error Rate (WER).
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)

def classification_metrics(y_true, y_pred):
    """
    Computes standard classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
