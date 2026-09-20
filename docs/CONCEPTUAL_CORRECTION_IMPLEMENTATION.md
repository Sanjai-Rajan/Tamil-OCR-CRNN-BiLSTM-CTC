# Deterministic Conceptual Correction Implementation

## 1. Motivation
The OCR model performs character-level recognition, which can sometimes produce words with minor spelling errors (e.g., swapping visually similar characters) due to noise or limited resolution. To correct these errors, a post-OCR correction module is needed. However, to maintain transparency, predictability, and to avoid data leakage or hallucination, this module must be purely deterministic and avoid any external neural networks or large language models (LLMs).

## 2. Existing Tamil Textual Resources
The project contains approximately 1,347 classical Tamil text documents (e.g., `.txt` files) located in `data/text_corpus/tamil/classical/project_madurai`. These files form a large, unannotated corpus of valid Tamil text.

## 3. Vocabulary Construction
A custom Python script scanned all 1,347 documents in the corpus, extracted all contiguous sequences of Tamil Unicode characters (`U+0B80` to `U+0BFF`), and applied NFKC normalization. The script extracted 2,801,994 total word tokens. 

To filter out noise, typos, and OCR artifacts present in the corpus, only words appearing at least 3 times were retained. This resulted in a clean vocabulary of **522,495 unique Tamil words**, saved as a JSON frequency map in `data/text_corpus/tamil_correction_vocabulary.json`.

## 4. Unicode Normalization
All incoming OCR predictions and corpus texts are normalized using NFKC (Normalization Form Compatibility Composition). This ensures that visual equivalents (like separate vowel markers vs pre-composed characters) are treated identically by the string matching algorithm.

## 5. Candidate Generation
When an OCR word is missing from the vocabulary, the corrector generates a set of candidates using exactly **one edit distance** (insertion, deletion, substitution, or transposition of any valid Tamil character).

## 6. Edit-Distance Logic
By restricting the edit distance to exactly 1, the algorithm guarantees it will not perform unbounded fuzzy matching. This prevents the system from drastically altering a word just to force it into the vocabulary.

## 7. Conservative Threshold
After generating candidates and finding which ones exist in the vocabulary, the algorithm sorts them by their corpus frequency. A candidate is only accepted if it has a frequency of at least **10** in the corpus. This acts as a strong evidence threshold.

## 8. Fallback Behavior
If no valid edit-distance 1 candidate is found in the vocabulary, or if the best candidate fails the frequency threshold, the corrector **gracefully returns the original OCR text**. It will not hallucinate or guess arbitrary words.

## 9. Integration Point
The corrector is integrated in `app/inference.py` inside the `OCRService.predict()` method. After a word is predicted by the CRNN and CTC-decoded, the string is passed to `DeterministicCorrector.correct()`. The corrected word is then appended to the line output. The service also tracks and reports the number of corrections made per image.

## 10. No Pretrained Language Model
This implementation relies strictly on `dict` lookups and set comprehensions in Python. It does not use any Transformer, RNN, or statistical n-gram library.

## 11. No Generative AI
There is no generative capacity. The system can only select words that already exist in the derived JSON vocabulary.

## 12. Limitations
- Words requiring two or more character edits will not be corrected.
- Rare but correct words (frequency < 10) may not be used as corrections, even if they are 1 edit away.
- Context is ignored; the corrector operates purely on isolated words (unigrams), so it cannot resolve grammatical ambiguities where both options are valid words.

## 13. Data Leakage Considerations
Because the vocabulary is derived from a classical text corpus (Project Madurai), there is a chance that some phrases or vocabulary words overlap with the historical manuscripts used in the OCR TEST dataset. Any performance gains on the TEST set resulting from this module should be explicitly categorized as "post-processing vocabulary improvements" rather than raw optical recognition capabilities.
