from typing import List, Dict, Any

class OCRUncertaintyDetector:
    def __init__(self, char_conf_threshold: float = 0.5, seq_conf_threshold: float = 0.7):
        # STATUS: INITIAL / UNCALIBRATED
        self.char_conf_threshold = char_conf_threshold
        self.seq_conf_threshold = seq_conf_threshold
        self.suspicious_chars = ["_", "?", ""]

    def analyze(self, ocr_text: str, character_confidences: List[float]) -> Dict[str, Any]:
        """
        Analyzes OCR output for uncertainty.
        Distinguishes visual/OCR uncertainty from linguistic errors.
        """
        if not ocr_text or not character_confidences:
            return {"needs_restoration": False, "reason": "empty_input"}

        # Calculate average sequence confidence (ignoring blanks if pre-filtered, else just raw mean)
        seq_conf = sum(character_confidences) / len(character_confidences)

        # Check thresholds
        if seq_conf < self.seq_conf_threshold:
            return {"needs_restoration": True, "reason": "low_sequence_confidence", "score": seq_conf}

        low_conf_chars = [c for c in character_confidences if c < self.char_conf_threshold]
        if len(low_conf_chars) > len(character_confidences) * 0.3:
            return {"needs_restoration": True, "reason": "many_low_confidence_characters", "count": len(low_conf_chars)}

        # Check for suspicious patterns
        for char in self.suspicious_chars:
            if char in ocr_text:
                return {"needs_restoration": True, "reason": f"suspicious_character_'{char}'"}

        return {"needs_restoration": False, "reason": "confident"}
