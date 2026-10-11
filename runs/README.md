# runs/

Each folder holds the frozen training configuration (`config.yaml`) of one documented experiment. These configurations are pinned against `configs/` by `tests/test_pretrain_check.py`.

The trained weights (`best.pth`) are **not included**. Put an authorised copy at `runs/<experiment>/best.pth` if you have one; see `docs/MODEL_CHECKPOINTS.md`. Weight files are ignored by `.gitignore`.
