# Interleaved Training Plan

## Objective
Implement a multi-objective block-training controller to safely switch between Masked Language Modeling (MLM), Character/Word Restoration, and Optical Character Recognition (OCR) fine-tuning phases.

## Cycle Structure
The training will follow a deterministic sequential cycle to prevent catastrophic forgetting while leveraging the architectural similarities:

**CYCLE DEFINITION:**
1. **MLM Block** (Domain adaptation and language structure)
2. **Restoration Block** (Contextual error correction and synthesis)
3. **OCR Block** (Visual-text alignment and transcription mapping)

## Block Configurations
- **MLM Block Size**: 2,000 optimizer steps
- **Restoration Block Size**: 1,000 optimizer steps
- **OCR Block Size**: 1,000 optimizer steps

*Note: Based on the current safe-stop of the initial MLM run at step 8,000, Cycle 1 will execute:*
- **MLM**: Step 8,000 → Step 10,000
- **Restoration**: Step 0 → Step 1,000
- **OCR**: Step 0 → Step 1,000

## Frequency & Metrics
- **Checkpoint Frequency**: Every 1,000 steps (and strictly enforced at block boundaries).
- **Evaluation Frequency**: Every 1,000 steps.

## Runtime Behaviors
- **Resume Behavior**: Upon launch, each block script must dynamically query its specific `CHECKPOINT_MANIFEST.csv` to identify the latest verified checkpoint, and instantiate the `Trainer` with `resume_from_checkpoint=latest`. 
- **Safe-Stop Behavior**: Instead of infinite running, each script will be configured with `max_steps` equal to its block target. The HuggingFace `Trainer` will naturally exit cleanly and save a final block checkpoint upon reaching `max_steps`.

## RTX 4060 ↔ RTX 5090 Transfer Procedure
1. Pause the interleaved controller at a cycle boundary.
2. Ensure no python training processes are active.
3. Synchronize `outputs/training/` and `checkpoints/` directories across machines.
4. Execute the verification script to cross-check all files against their respective `CHECKPOINT_MANIFEST.csv` SHA-256 hashes.
5. Once `VERIFIED`, start the controller on the target machine. It will automatically detect the block boundaries and resume seamlessly.
