import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="A100 Experiment Runner")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    print(f"=== Starting A100 Experiment: {args.experiment} ===")
    
    cmd = [
        sys.executable, "scripts/train_crnn.py",
        "--experiment", args.experiment,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", args.device
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
        
    print(f"Executing: {' '.join(cmd)}")
    
    # Do NOT execute training as per current instruction.
    print("Runner ready. Execution is blocked by current safety constraints.")
    print("Exiting.")

if __name__ == "__main__":
    main()
