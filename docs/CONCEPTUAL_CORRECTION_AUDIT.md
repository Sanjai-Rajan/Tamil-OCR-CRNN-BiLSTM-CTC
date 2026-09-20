# Conceptual Correction Audit

## 1. Objective
This audit inspects the repository to determine the exact state of the deterministic conceptual/text correction module for the Tamil OCR pipeline.

## 2. Audit Findings

**State:** **C. Deterministic Post-Processing**

### Evidence:
1. **app/inference.py**: The inference pipeline uses the raw CTC decode from the CRNN, but now passes the output through `self.corrector.correct()` before reconstructing the final lines.
2. **models/digitalization/conceptual_corrector.py**: Contains `DeterministicCorrector`, which generates edit-distance 1 candidates and matches them against a vocabulary.
3. **data/text_corpus/tamil_correction_vocabulary.json**: A frequency vocabulary of 522,495 unique Tamil words derived from the classical Project Madurai corpus.

## 3. Detailed Inspection
- **Madurai/Tamil text resources**: Present (`data/text_corpus/tamil_correction_vocabulary.json`).
- **Vocabulary resources**: Present (522,495 unique Tamil words).
- **Correction rules**: Present (Edit-distance 1 with strict corpus frequency threshold of 10).
- **Vocabulary matching**: Present (Exact match fallback with candidate generation).
- **Post-OCR processing**: Present. Deterministic post-processing is applied to OCR outputs with a safe fallback to raw OCR text.
- **Existing correction code**: Present (`models/digitalization/conceptual_corrector.py`).

## 4. Constraint Verification
- **AI/LLM Usage**: **PASS**. No external AI language models, mT5, or transformers are integrated into the pipeline. The repository relies strictly on the CRNN model and deterministic post-processing.

## 5. Conclusion
The conceptual correction module is **present** in the repository as **State C (Deterministic Post-Processing)**. The OCR pipeline incorporates a deterministic vocabulary and post-processing heuristics derived from the classical Tamil corpus (Project Madurai) to correct character-level errors without relying on external AI language models, transformers, or unrestricted fuzzy matching.
