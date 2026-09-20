# Error Log

## Resolved Errors

### Error 1
**ModuleNotFoundError:**
`No module named "recognition"`

**Resolution:**
* Added project path handling.
* Fixed package-based imports.
* Added/fixed `__init__.py` files.
* Fixed imports in CRNN-related modules.

### Error 2
**RuntimeError:**
`Input type unsigned char and bias type float should be the same.`

**Resolution:**
* Convert image arrays to `torch.float32`.
* Normalize pixel values appropriately.

### Error 3
**RuntimeError:**
`input.size(-1) must be equal to input_size. Expected 512, received 8192.`

**Resolution:**
* CRNN sequence dimensions were corrected.
* LSTM input size was adjusted.
* Dynamic feature dimensions should be preferred over unnecessary hard-coded dimensions.

### Error 4
**ImportError:**
`cannot import name 'CTCGreedyDecoder'`

**Resolution:**
* The decoder currently uses the functional `ctc_decode` approach.
* Existing tests were updated to target the actual decoder implementation.

IMPORTANT: Before changing any recognition code, inspect the CURRENT implementation and preserve these fixes.
