import torch
import torch.nn.functional as F

def ctc_decode(predictions):
    predictions = torch.argmax(
        predictions,
        dim=2
    )
    return predictions

def ctc_decode_with_confidence(predictions):
    """
    Decodes CTC predictions and returns character-level confidences.
    Args:
        predictions: Logits from CRNN [batch, seq_len, num_classes]
    Returns:
        decoded: Extracted class indices [batch, seq_len]
        confidences: List of lists containing float probabilities [batch, seq_len]
    """
    probs = F.softmax(predictions, dim=2)
    max_probs, decoded = torch.max(probs, dim=2)
    return decoded, max_probs.tolist()
