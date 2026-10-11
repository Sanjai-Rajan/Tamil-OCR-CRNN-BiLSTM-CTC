# Experiment 001 Config Mapping Report

This report documents exactly how the configuration in `configs/experiment_001.yaml` satisfies the approved design from `reports/experiment_001_design_review.md` and incorporates the prerequisites and safety fixes.

## Configuration Mapping

1. **Name and Isolation:**
   - **Config:** `experiment: experiment_001`
   - **Mapping:** Ensures the outputs are written to `runs/experiment_001/`. The baseline (`runs/baseline_hw/`) is entirely untouched and unreferenced. 

2. **Model Initialization Constraint:**
   - **Mapping:** No `--resume` or fine-tuning weights are specified in the config. `src/train.py` builds the CRNNv2 model from scratch because no loading path is provided.

3. **Data Sources and Mixing:**
   - **Handwriting Source:**
     - `name: iiit_hw`
     - `path: data/real/iiit_hw/manifest_train.jsonl`
     - `augment: hw`
     - `weight: 0.5`
   - **Synthetic Source:**
     - `name: synth_print`
     - `type: synth`
     - `vocab: data/text_corpus/tamil_correction_vocabulary.json`
     - `old_print_prob: 0.5`
     - `weight: 0.5`
   - **Mapping:** As required in §6.2 of the design review, this enforces exactly equal expected proportions of handwriting and synthetic printed text. The `old_print_prob: 0.5` ensures that the synthetic samples are drawn in the exact 50/50 modern/old distribution described in the design.

4. **Epoch Size and Handwriting Exposure:**
   - **Config:** `epoch_size: 151472`
   - **Mapping:** The baseline used `75736` at `1.0` weight. By doubling the epoch size to `151472` with a `0.5` handwriting weight, the expected number of handwriting samples drawn per epoch remains exactly `75736`, preserving handwriting exposure while introducing the synthetic data.

5. **Safety Fixes Integration:**
   - **Config:** `unknown_width_estimate: 517`
   - **Mapping:** Mitigates the synthetic width explosion (where estimates originally defaulted to `None` and fell back to the handwriting median). We measured a max synthetic width of 517px, which safely anchors the `WidthBucketSampler` logic without modifying `max_batch_columns: 32768`.

6. **Validation Targets:**
   - **Config:**
     - `iiit_hw_val` (`primary: true`)
     - `synth_val_modern` (`where: {style: modern}`, `primary: false`)
     - `synth_val_old` (`where: {style: old}`, `primary: false`)
   - **Mapping:** The IIIT handwritten validation set remains the sole `primary` selection signal for checkpointing, while the modern and old synthetic subsets are strictly monitored without influencing model selection.

7. **Other Constant Factors:**
   - All other hyperparameters (`seed: 42`, `model` architecture, `lr: 0.001`, `max_batch_columns: 32768`, `fused_adamw: true`, `amp: true`, `epochs: 40`, `max_hours: 24`, etc.) are exact copies of `configs/baseline_hw.yaml`.
