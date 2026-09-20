import sys
import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import csv
import time
import gc
from collections import Counter

# Force UTF-8 encoding for standard output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))



import torch.nn as nn
from models.recognition.lstm import SequenceModel
from models.recognition.head import PredictionHead

class EncoderW2(nn.Module):
    def __init__(self, less_downsample=False):
        super().__init__()
        # Preserving W/2 resolution horizontally:
        pool1 = nn.MaxPool2d((2, 2))
        pool2 = nn.MaxPool2d((2, 1)) # Changed from (2,2)
        pool3 = nn.MaxPool2d((2, 1))

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            pool1,

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            pool2,

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            pool3,

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)

class CRNNW2(nn.Module):
    def __init__(self, num_classes, feature_size=8192, less_downsample=False):
        super().__init__()
        self.encoder = EncoderW2(less_downsample=less_downsample)
        # SequenceModel input_size remains the same (512 * 8 = 4096 depending on height? 
        # Wait, if height downsamples by 2*2*2=8, input height is 32, so 32/8=4. 
        # channel=512, 512*4=2048. SequenceModel input_size=feature_size which defaults to 2048 in training call.
        self.sequence = SequenceModel(input_size=feature_size)
        self.head = PredictionHead(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        batch, channel, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, width, channel * height)
        x = self.sequence(x)
        x = self.head(x)
        return x

from models.recognition.decoder import ctc_decode
from scripts.dataset_loader import get_dataloader
from models.digitalization.tokenizer import Tokenizer
from utils.device import get_device
from utils.metrics import calculate_cer, calculate_wer

def setup_argparse():
    parser = argparse.ArgumentParser(description="Train CRNN for OCR (Staged)")
    parser.add_argument("--language", type=str, default="tamil", help="Language to train on")
    parser.add_argument("--packet", type=str, default=None, help="Target specific packet (e.g., packet_001)")
    parser.add_argument("--epochs", type=int, default=10, help="NUMBER OF ADDITIONAL EPOCHS TO TRAIN.")
    parser.add_argument("--total-epochs", type=int, default=10, help="TOTAL epochs for the experiment (for correct OneCycleLR schedule)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--max-packets", type=int, default=None, help="Max packets to load")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers for dataloader")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer steps)")
    parser.add_argument("--experiment", type=str, default="ocr_temporal_resolution_w2", help="Experiment subfolder for outputs and checkpoints")
    parser.add_argument("--exclude-packet", type=str, default=None, help="Packet to exclude from training")
    return parser

def load_config(config_path="config/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

import random
import numpy as np

def save_checkpoint(model, optimizer, scheduler, state, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "rng_state": rng_state,
        "state": state,
        "config": config
    }
    torch.save(checkpoint, path)

def run_validation(model, dataloader, criterion, tokenizer, device, debug=False):
    model.eval()
    val_loss = 0
    total_cer = 0
    total_wer = 0
    num_samples = 0
    
    examples = []
    
    all_predicted_chars = []
    unique_words = set()
    total_raw_preds = 0
    blank_preds = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 5:
                images, targets, target_lengths, actual_widths, image_paths = batch_data
            else:
                images, targets, target_lengths, actual_widths = batch_data
            
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            actual_widths = actual_widths.to(device, non_blocking=True)
            
            outputs = model(images)
            outputs_perm = outputs.permute(1, 0, 2)
            outputs_log_probs = outputs_perm.log_softmax(2)
            
            input_lengths = torch.round((actual_widths.float() / images.size(3)) * outputs_perm.size(0)).to(torch.long)
            input_lengths = torch.clamp(input_lengths, max=outputs_perm.size(0))
            
            loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
            val_loss += loss.item()
            
            preds = ctc_decode(outputs)
            
            total_raw_preds += preds.numel()
            blank_preds += (preds == 0).sum().item()
            
            offset = 0
            for i in range(len(target_lengths)):
                length = target_lengths[i].item()
                target_seq = targets[offset:offset+length].cpu().tolist()
                offset += length
                
                try:
                    ref_text = tokenizer.decode(target_seq)
                except AttributeError:
                    ref_text = "".join([str(c) for c in target_seq])
                    
                pred_seq = preds[i].cpu().tolist()
                clean_pred = []
                prev = -1
                for p in pred_seq:
                    if p != 0 and p != prev:
                        clean_pred.append(p)
                    prev = p
                try:
                    hyp_text = tokenizer.decode(clean_pred)
                except AttributeError:
                    hyp_text = "".join([str(c) for c in clean_pred])
                    
                for c in hyp_text:
                    all_predicted_chars.append(c)
                unique_words.add(hyp_text)
                    
                cer = calculate_cer(ref_text, hyp_text)
                total_cer += cer
                total_wer += calculate_wer(ref_text, hyp_text)
                
                if len(examples) < 10:
                    examples.append({
                        "gt": ref_text,
                        "pred": hyp_text,
                        "cer": cer
                    })
                    
                num_samples += 1
                
            if debug and batch_idx >= 1:
                break
                
    if num_samples == 0:
        return 0, 0, 0, examples, {}
        
    char_counts = Counter(all_predicted_chars)
    total_chars_predicted = sum(char_counts.values())
    if total_chars_predicted > 0:
        most_common_char, count = char_counts.most_common(1)[0]
        dom_percentage = (count / total_chars_predicted) * 100
    else:
        most_common_char = ""
        dom_percentage = 0.0
        
    mode_collapse_stats = {
        "unique_characters": len(char_counts),
        "unique_words": len(unique_words),
        "most_common_char": most_common_char,
        "dominance_percentage": dom_percentage,
        "blank_percentage": (blank_preds / total_raw_preds * 100) if total_raw_preds > 0 else 100.0,
        "avg_pred_length": (total_chars_predicted / num_samples) if num_samples > 0 else 0.0
    }
        
    return val_loss / (batch_idx + 1), total_cer / num_samples, total_wer / num_samples, examples, mode_collapse_stats

def verify_one_batch(model, dataloader, criterion, tokenizer, device):
    """
    Verifies a single batch before training.
    """
    print("============================================================")
    print("VERIFYING ONE BATCH BEFORE TRAINING")
    print("============================================================")
    
    model.train()
    
    try:
        batch_data = next(iter(dataloader))
        if len(batch_data) == 5:
            images, targets, target_lengths, actual_widths, image_paths = batch_data
        else:
            images, targets, target_lengths, actual_widths = batch_data
    except StopIteration:
        print("ERROR: Dataloader is completely empty.")
        sys.exit(1)
        
    print(f"Images shape: {images.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Target lengths shape: {target_lengths.shape}")
    print(f"Actual widths shape: {actual_widths.shape}")
    if len(batch_data) == 5:
        packet_names = set([Path(p).parent.parent.name for p in image_paths])
        print(f"Number of unique packets in this batch: {len(packet_names)}")
    
    if len(targets) == 0:
        print("ERROR: Labels are completely empty.")
        sys.exit(1)
        
    images = images.to(device)
    targets = targets.to(device)
    
    outputs = model(images)
    print(f"Outputs shape: {outputs.shape}")
    
    outputs_perm = outputs.permute(1, 0, 2)
    outputs_log_probs = outputs_perm.float().log_softmax(2)
    
    input_lengths = torch.round((actual_widths.float() / images.size(3)) * outputs_perm.size(0)).to(torch.long)
    input_lengths = torch.clamp(input_lengths, min=1, max=outputs_perm.size(0))
    
    loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {loss.item():.4f}")
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("ERROR: CTC Loss is NaN or Inf.")
        sys.exit(1)
        
    if loss.item() < 0:
        print("ERROR: CTC Loss is negative. Log_softmax might be failing.")
        sys.exit(1)
        
    first_target_len = target_lengths[0].item()
    first_target_seq = targets[:first_target_len].cpu().tolist()
    decoded = tokenizer.decode(first_target_seq)
    print(f"First label decoded: '{decoded}'")
    
    if not decoded:
        print("ERROR: First label is empty after decoding. Tokenizer failed.")
        sys.exit(1)
        
    print("Verification PASSED. Starting training loop...\n")

def train():
    parser = setup_argparse()
    args = parser.parse_args()
    config = load_config()
    
    epochs_to_add = args.epochs
    batch_size = args.batch_size or config["training"].get("batch_size", 16)
    lr = args.lr or config["training"].get("learning_rate", 0.001)
    
    num_workers_conf = config["training"].get("num_workers", 0)
    num_workers = args.num_workers if args.num_workers is not None else num_workers_conf
    
    working_root = config["dataset"]["working_root"]
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
        
    print("============================================================")
    print("HARDWARE DETECTED")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: None")
    print(f"PyTorch version: {torch.__version__}")
    print("============================================================\n")
    
    vocab_path = os.path.join(working_root, "vocabulary", f"{args.language}_vocab.json")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    
    model = CRNNW2(tokenizer.num_classes, feature_size=2048, less_downsample=True).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=config["training"].get("weight_decay", 0.0001))
    
    # Calculate total steps for OneCycleLR
    estimated_steps_per_epoch = 126419 // batch_size + 1
    total_epochs = args.total_epochs
    total_steps = estimated_steps_per_epoch * total_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 5, total_steps=total_steps, 
        pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
    )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    if args.experiment:
        outputs_dir = Path("outputs") / "training" / args.language / args.experiment
        ckpt_dir = Path(config["checkpoint"].get("directory", f"checkpoints/recognition/{args.language}")) / args.experiment
    else:
        outputs_dir = Path("outputs") / "training" / args.language
        ckpt_dir = Path(config["checkpoint"].get("directory", f"checkpoints/recognition/{args.language}"))
        
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    history_csv = outputs_dir / "training_history.csv"
    history_json = outputs_dir / "training_history.json"
    state_json_path = outputs_dir / "training_state.json"
    
    training_state = {
        "language": args.language,
        "global_epoch": 1,
        "last_completed_packet": None,
        "best_validation_loss": float('inf'),
        "best_cer": float('inf'),
        "best_wer": float('inf')
    }
    
    start_epoch = 1
    checkpoint_loaded = None
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        if not os.path.exists(args.resume):
            print(f"ERROR: Checkpoint not found at {args.resume}")
            sys.exit(1)
            
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        if "state" in checkpoint:
            training_state = checkpoint["state"]
            # The checkpoint's global_epoch is the last completed epoch
            start_epoch = training_state.get("global_epoch", 0) + 1
            
        print(f"Restored model from epoch {start_epoch}. Last completed packet: {training_state.get('last_completed_packet', 'None')}")
        checkpoint_loaded = checkpoint
    elif state_json_path.exists():
        # Auto-resume from state file if no checkpoint explicitly provided but state exists in this experiment
        print(f"Found existing training_state.json in {outputs_dir}. Loading state...")
        with open(state_json_path, 'r', encoding='utf-8') as f:
            saved_state = json.load(f)
            training_state.update(saved_state)
            # The state's global_epoch is the last completed epoch
            start_epoch = training_state.get("global_epoch", 0) + 1
        latest_path = ckpt_dir / "latest.pth"
        if latest_path.exists():
            print(f"Auto-resuming weights from {latest_path}...")
            checkpoint = torch.load(latest_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            checkpoint_loaded = checkpoint
            
    if checkpoint_loaded:
        if "scheduler_state_dict" in checkpoint_loaded and checkpoint_loaded["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint_loaded["scheduler_state_dict"])
            print("Restored exact scheduler state.")
        else:
            # Reconstruct legacy scheduler state
            print("Legacy checkpoint detected. Reconstructing OneCycleLR state for end of Epoch 1...")
            steps_to_advance = estimated_steps_per_epoch * (start_epoch - 1)
            for _ in range(steps_to_advance):
                scheduler.step()
            print(f"Advanced newly created scheduler by {steps_to_advance} steps.")
            
        if "rng_state" in checkpoint_loaded and checkpoint_loaded["rng_state"] is not None:
            rng = checkpoint_loaded["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            try:
                cpu_rng = rng["torch_cpu"]
                if not isinstance(cpu_rng, torch.Tensor):
                    cpu_rng = torch.tensor(cpu_rng, dtype=torch.uint8)
                torch.set_rng_state(cpu_rng.byte())
                if rng["torch_cuda"] is not None and torch.cuda.is_available():
                    cuda_rngs = [torch.tensor(r, dtype=torch.uint8).byte() if not isinstance(r, torch.Tensor) else r.byte() for r in rng["torch_cuda"]]
                    torch.cuda.set_rng_state_all(cuda_rngs)
                print("Restored exact RNG states.")
            except Exception as e:
                print(f"Warning: exact RNG restoration skipped due to error: {e}")
        else:
            print("Legacy checkpoint: exact RNG restoration skipped. Using fresh RNG state.")
    
    end_epoch = start_epoch + epochs_to_add
    
    # Discover available packets
    imported_dir = Path(working_root) / "imported" / args.language / "train"
    if not imported_dir.exists():
        print(f"ERROR: Imported training directory not found at {imported_dir}")
        sys.exit(1)
        
    all_packets = sorted([d.name for d in imported_dir.iterdir() if d.is_dir() and d.name.startswith("packet_")])
    
    if args.packet:
        if args.packet in all_packets:
            all_packets = [args.packet]
        else:
            print(f"ERROR: Target packet {args.packet} not found in {imported_dir}")
            sys.exit(1)
            
    if args.max_packets:
        all_packets = all_packets[:args.max_packets]
        
    if not all_packets:
        print("ERROR: No imported packets found.")
        sys.exit(1)
        
    # Pre-load validation dataset (evaluate against all validation packets)
    val_loader = get_dataloader(
        dataset_root=working_root, 
        language=args.language, 
        split="validation", 
        tokenizer=tokenizer, 
        batch_size=batch_size, 
        num_workers=num_workers,
        max_packets=None, # Load all validation data
        target_packet=None,
        replay_previous=False,
        shuffle=False
    )
    
    history_records = []
    if history_json.exists():
        with open(history_json, 'r', encoding='utf-8') as f:
            try:
                history_records = json.load(f)
            except json.JSONDecodeError:
                pass
                
    scaler = torch.cuda.amp.GradScaler() if (config["training"].get("mixed_precision", True) and device.type == 'cuda') else None

    # Determine start idx for the current epoch
    last_completed = training_state.get("last_completed_packet")
    packet_start_idx = 0
    if last_completed and last_completed in all_packets:
        packet_start_idx = all_packets.index(last_completed) + 1
        if packet_start_idx >= len(all_packets):
            print(f"Epoch {start_epoch} already completed all packets. Moving to next epoch.")
            start_epoch += 1
            packet_start_idx = 0

    first_batch_verified = False

    try:
        for epoch in range(start_epoch, end_epoch):
            print(f"\n============================================================")
            print(f"Epoch {epoch}/{end_epoch-1} | Global Shuffling Experiment")
            print(f"============================================================")
            
            # 1. Load GLOBAL dataset (ALL packets)
            train_loader = get_dataloader(
                dataset_root=working_root, 
                language=args.language, 
                split="train", 
                tokenizer=tokenizer, 
                batch_size=batch_size, 
                num_workers=num_workers,
                max_packets=None,
                target_packet=None, # This ensures ALL packets are loaded
                replay_previous=False
            )
            
            if args.exclude_packet and train_loader is not None:
                train_loader.dataset.data = [
                    item for item in train_loader.dataset.data
                    if args.exclude_packet not in Path(item["image_path"]).parts
                ]
                print(f"Excluded training packet: {args.exclude_packet}")
                print(f"Training samples after exclusion: {len(train_loader.dataset.data)}")

            if train_loader is None or len(train_loader) == 0:
                print(f"ERROR: Global dataloader is empty.")
                sys.exit(1)
                
            total_samples = len(train_loader.dataset)
            num_packets = len(all_packets)
            print(f"Total Packets: {num_packets}")
            print(f"Total Training Samples: {total_samples}")
            print(f"Batch Size: {batch_size}")
            print(f"Number of Batches: {len(train_loader)}")
            print(f"Shuffle: True")
            print(f"Experiment Name: {args.experiment or 'None'}")
            print("============================================================\n")
                
            if not first_batch_verified:
                verify_one_batch(model, train_loader, criterion, tokenizer, device)
                first_batch_verified = True
                
            model.train()
            epoch_loss = 0
            num_batches = 0
            skipped_batches = 0
            total_batches = 0
            packet_start_time = time.time()
            
            # 2. Train for one pass through the global dataset
            for batch_idx, batch_data in enumerate(train_loader):
                if len(batch_data) == 5:
                    images, targets, target_lengths, actual_widths, image_paths = batch_data
                else:
                    images, targets, target_lengths, actual_widths = batch_data
                    
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                actual_widths = actual_widths.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                total_batches += 1
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        outputs = outputs.permute(1, 0, 2)
                        outputs_log_probs = outputs.float().log_softmax(2)
                        
                        input_lengths = torch.round((actual_widths.float() / images.size(3)) * outputs.size(0)).to(torch.long)
                        input_lengths = torch.clamp(input_lengths, min=1, max=outputs.size(0))
                        
                        loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
                    
                    if not torch.isfinite(loss):
                        print(f"  [WARNING] Epoch {epoch} Batch {batch_idx+1}: NaN/Inf loss detected. Skipping.")
                        skipped_batches += 1
                        continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    outputs = model(images)
                    outputs = outputs.permute(1, 0, 2)
                    outputs_log_probs = outputs.float().log_softmax(2)
                    
                    input_lengths = torch.round((actual_widths.float() / images.size(3)) * outputs.size(0)).to(torch.long)
                    input_lengths = torch.clamp(input_lengths, min=1, max=outputs.size(0))
                    
                    loss = criterion(outputs_log_probs, targets, input_lengths, target_lengths)
                    
                    if not torch.isfinite(loss):
                        print(f"  [WARNING] Epoch {epoch} Batch {batch_idx+1}: NaN/Inf loss detected. Skipping.")
                        skipped_batches += 1
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    elapsed = time.time() - packet_start_time
                    speed = (num_batches * batch_size) / elapsed if elapsed > 0 else 0
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Samples: {num_batches * batch_size} | Time: {elapsed:.1f}s | Speed: {speed:.1f} samples/s")
                    sys.stdout.flush()
                    
                if args.debug and batch_idx >= 1:
                    break
                    
            epoch_duration = time.time() - packet_start_time
            avg_train_loss = epoch_loss / max(1, num_batches)
            
            # --- PRE-VALIDATION CHECKPOINT ---
            print(f"\n[SAVE] Saving pre-validation checkpoint for epoch {epoch}...")
            pre_val_path = str(ckpt_dir / f"epoch_{epoch:03d}_pre_validation.pth")
            training_state["global_epoch"] = epoch
            save_checkpoint(model, optimizer, scheduler, training_state, config, pre_val_path)
            print(f"[SAVE] Pre-validation checkpoint saved to {pre_val_path}")
            
            # 3. Validate
            val_loss, cer, wer, examples, mc_stats = 0, 0, 0, [], {}
            if val_loader:
                val_loss, cer, wer, examples, mc_stats = run_validation(model, val_loader, criterion, tokenizer, device, debug=args.debug)
            
            char_acc = max(0, 1.0 - cer)
            word_acc = max(0, 1.0 - wer)
            
            # 4. Calculate metrics
            print(f"\n--- Epoch {epoch} Metrics ---")
            print(f"Total Batches Processed: {total_batches}")
            print(f"Skipped NaN/Inf Batches: {skipped_batches}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"CER: {cer:.4f}")
            print(f"WER: {wer:.4f}")
            print(f"Character Accuracy: {char_acc:.4f}")
            print(f"Word Accuracy: {word_acc:.4f}")
            
            print(f"\n--- Mode Collapse Analysis ---")
            print(f"Blank Prediction %:          {mc_stats.get('blank_percentage', 100):.2f}%")
            print(f"Unique Predicted Characters: {mc_stats.get('unique_characters', 0)}")
            print(f"Unique Predicted Sequences:  {mc_stats.get('unique_words', 0)}")
            print(f"Dominant Character:          '{mc_stats.get('most_common_char', '')}' ({mc_stats.get('dominance_percentage', 0):.2f}%)")
            print(f"Average Prediction Length:   {mc_stats.get('avg_pred_length', 0):.2f}")
            print(f"Training Time:               {epoch_duration:.1f}s")
            print(f"Samples/sec:                 {(total_samples/epoch_duration):.1f}")
            print(f"GPU/Device:                  {device}")
            
            # Show 10 validation predictions
            print(f"\n--- Validation Predictions ---")
            for ex in examples:
                print(f"GT: {ex['gt']} | Pred: {ex['pred']}")
                
            sys.stdout.flush()
            
            # Update state
            training_state["global_epoch"] = epoch
            
            # Update best metrics
            is_best = False
            if cer < training_state.get("best_cer", float('inf')):
                training_state["best_cer"] = cer
                training_state["best_wer"] = wer
                training_state["best_validation_loss"] = val_loss
                is_best = True
            
            # 5. Save Checkpoint
            latest_path = str(ckpt_dir / "latest.pth")
            epoch_path = str(ckpt_dir / f"epoch_{epoch:03d}.pth")
            
            save_checkpoint(model, optimizer, scheduler, training_state, config, latest_path)
            save_checkpoint(model, optimizer, scheduler, training_state, config, epoch_path)
            
            if is_best:
                best_path = str(ckpt_dir / "best.pth")
                save_checkpoint(model, optimizer, scheduler, training_state, config, best_path)
                print(f"[SAVE] New best CER ({cer:.4f}), saved to best.pth")
            
            with open(state_json_path, 'w', encoding='utf-8') as f:
                json.dump(training_state, f, indent=4)
                
            # Create Markdown report
            report_path = outputs_dir / f"epoch_{epoch:03d}_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# Epoch {epoch} Report\n\n")
                f.write(f"- Train Loss: {avg_train_loss:.4f}\n")
                f.write(f"- Validation Loss: {val_loss:.4f}\n")
                f.write(f"- CER: {cer:.4f}\n")
                f.write(f"- WER: {wer:.4f}\n")
                f.write(f"- Character Accuracy: {char_acc:.4f}\n")
                f.write(f"- Word Accuracy: {word_acc:.4f}\n")
                f.write(f"- Blank Prediction %: {mc_stats.get('blank_percentage', 100):.2f}%\n")
                f.write(f"- Unique Predicted Characters: {mc_stats.get('unique_characters', 0)}\n")
                f.write(f"- Unique Predicted Sequences: {mc_stats.get('unique_words', 0)}\n")
                f.write(f"- Average Prediction Length: {mc_stats.get('avg_pred_length', 0):.2f}\n")
                f.write(f"- GPU/Device: {device}\n")
                f.write(f"- Training Time: {epoch_duration:.1f}s\n")
                f.write(f"- Samples/sec: {(total_samples/epoch_duration):.1f}\n")
                f.write(f"\n## Validation Examples\n\n")
                for ex in examples:
                    f.write(f"- **GT**: {ex['gt']}\n  **Pred**: {ex['pred']}\n\n")

            # Metrics dict for CSV/JSON
            metrics = {
                "epoch": epoch,
                "samples": total_samples,
                "train_loss": avg_train_loss,
                "validation_loss": val_loss,
                "CER": cer,
                "WER": wer,
                "character_accuracy": char_acc,
                "word_accuracy": word_acc,
                "blank_percentage": mc_stats.get('blank_percentage', 100),
                "unique_predicted_characters": mc_stats.get('unique_characters', 0),
                "unique_predicted_sequences": mc_stats.get('unique_words', 0),
                "average_prediction_length": mc_stats.get('avg_pred_length', 0),
                "learning_rate": lr,
                "training_time": epoch_duration
            }
            
            history_records.append(metrics)
            
            with open(history_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                if not history_csv.exists() or os.path.getsize(history_csv) == 0:
                    writer.writeheader()
                writer.writerow(metrics)
                
            with open(history_json, 'w', encoding='utf-8') as f:
                json.dump(history_records, f, indent=4, ensure_ascii=False)
                
            # 6. Release dataloader
            del train_loader
            gc.collect()
            
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] KeyboardInterrupt detected!")
        print(f"Last completed packet: {training_state.get('last_completed_packet', 'None')}")
        try:
            print(f"Current batch: {batch_idx+1}/{len(train_loader)}")
        except NameError:
            pass
        print("Saving emergency state and exiting cleanly...")
        
        latest_path = str(ckpt_dir / "latest.pth")
        save_checkpoint(model, optimizer, scheduler, training_state, config, latest_path)
        with open(state_json_path, 'w', encoding='utf-8') as f:
            json.dump(training_state, f, indent=4)
            
        sys.exit(0)
        
    print("Training run complete.\n")

if __name__ == "__main__":
    train()
