# First Real MLM Checkpoint Audit

## Current Training State

**1. GLOBAL_STEP:** 200 (In progress towards 1000 for first checkpoint)
**2. CURRENT_EPOCH:** 0.0026
**3. TRAIN_LOSS:** 7.231
**4. VALIDATION_LOSS:** N/A (Awaiting eval at step 1000)
**5. PERPLEXITY:** N/A
**6. LEARNING_RATE:** ~2.00e-05
**7. TOKENS/EXAMPLES PROCESSED:** 6,400 examples (819,200 tokens)
**8. CHECKPOINT_PATH:** N/A (Will be created at step 1000)
**9. CHECKPOINT_COMPLETE:** NO
**10. CHECKPOINT_RESUME_STATE:** INVALID
**11. CHECKPOINT_MANIFEST:** MISSING
**12. SHA256_VERIFICATION:** FAIL (No checkpoint exists yet)
**13. GPU_MEMORY:** 7874MiB / 8188MiB (96% Utilization - Healthy)
**14. TRAINING_THROUGHPUT:** ~2.30 iterations/sec (~73.6 examples/sec)
**15. ESTIMATED_TIME_REMAINING:** ~9h 20m

## Checkpoint Contents Verification
Currently, the training directory (`checkpoints/mlm/indicbert_run_01/`) is empty because the script is configured to save checkpoints every 1000 steps. 

Once step 1000 is reached, the following resumable training state will be available:
- model weights (safetensors/bin)
- optimizer state (`optimizer.pt`)
- scheduler state (`scheduler.pt`)
- trainer state (`trainer_state.json`)
- RNG state (`rng_state.pth`)
- configuration (`config.json`, `training_args.bin`)
- tokenizer dependencies

## FINAL STATUS
**WAITING_FOR_MORE_TRAINING**
