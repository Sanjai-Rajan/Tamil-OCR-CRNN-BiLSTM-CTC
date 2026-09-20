# Pipeline Input/Output Contracts

## Image Preprocessing (`models.cv.preprocessor`)
*   **Input**: Raw `np.ndarray` (Image)
*   **Output**: Binarized, denoised `np.ndarray`

## Segmentation (`models.cv.segmenter`)
*   **Input**: Preprocessed `np.ndarray`
*   **Output**: List of bounding boxes or cropped `np.ndarray`s representing lines/words.

## Language Detection (`models.language_detection.language_detector`)
*   **Input**: Preprocessed `np.ndarray` or text chunk.
*   **Output**: `dict` containing `language`, `script`, `direction`, and `confidence`.
    *   *Example*: `{"language": "tamil", "direction": "ltr", "confidence": 0.98}`

## Recognition / OCR (`models.recognition.crnn`)
*   **Input**: Normalized `torch.Tensor` of shape `[Batch, Channels, Height, Width]`
*   **Output**: Raw text string transcription.

## Digitalization (`models.digitalization.sentence_builder`)
*   **Input**: Raw OCR text string.
*   **Output**: Structured JSON payload containing token-level confidences and statuses.
    *   *Example*: `{"tokens": [{"text": "அரண்ம_னை", "confidence": 0.51, "status": "uncertain"}]}`

## Reconstruction (`models.reconstruction.reconstructor`)
*   **Input**: Digitalized text output with gaps or uncertain tokens.
*   **Output**: Reconstructed text string with alternative candidates and validation status.

## Translation (`models.translation.translator`)
*   **Input**: Final reconstructed native-language string.
*   **Output**: English translated string.
