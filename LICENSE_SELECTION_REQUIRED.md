# Licence selection required before public release

This repository has **no licence file**. Without one, no rights to use, copy, modify or redistribute the project's own code and documentation are granted. The owner must decide on the licence; it was deliberately not chosen during the export.

## Decisions needed

1. **The project's own code and documentation:** choose a licence (for example MIT, Apache-2.0, BSD-3-Clause, or GPL-3.0 if derivatives should stay open), and confirm that the owner holds the rights to all of it, and that no institution, employer or programme terms (for example C-DAC) restrict publication.
2. **Trained weights**, which are not in this repository: whether, and under which terms, they may ever be shared. They were trained on a third-party handwriting dataset (IIIT-INDIC-HW-WORDS, Tamil), whose terms bear on derived weights. Check those terms first.
3. **The protected-data fingerprint file** `pageocr/protected_manifest.json`: sizes and SHA-256 hashes of the project's protected research files (no names, paths or content). Confirm that publishing these fingerprints is acceptable. Removing the file makes the application refuse every upload (the guard fails closed), so removal also needs a code decision.

## Already settled

- **Fonts** (`assets/fonts/`): 15 third-party families under the SIL Open Font License 1.1. Each family's licence text (`OFL_*.txt`) is next to its font files, and the OFL permits redistribution with that text.
  - The export copied the font files byte for byte; that they are unmodified upstream releases was not re-checked against their original sources.
  - The OFL keeps the font authors' copyright and reserved font names; the project's licence does not apply to the fonts.
- **No dataset or word list is included**, so no dataset licence is needed for the repository itself.
