import os
import sys
import time
import math
import psutil
import subprocess
from datetime import datetime
from pathlib import Path

# Paths
ROOT_DIR = Path(r"C:\Users\prsan\Desktop\CDAC\1_Draft")
RUN_DIR = ROOT_DIR / "outputs" / "training" / "mlm_run_01"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "mlm" / "indicbert_run_01"
PROGRESS_FILE = RUN_DIR / "TRAINING_PROGRESS.md"
CONFIG_FILE = RUN_DIR / "TRAINING_CONFIG.md"
LOG_FILE = RUN_DIR / "LIVE_MONITOR.log"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def log_event(message):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

# State variables
last_known_checkpoints = set()
total_epochs = 3 # default estimate

def parse_config():
    config = {
        "batch_size": "N/A",
        "grad_acc": "N/A",
        "eff_batch_size": "N/A",
        "lr": "N/A",
        "precision": "FP32",
        "max_length": "N/A"
    }
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if "Batch Size:" in line and "Effective" not in line:
                    config["batch_size"] = line.split(":")[-1].strip()
                elif "Gradient Accumulation" in line:
                    config["grad_acc"] = line.split(":")[-1].strip()
                elif "Effective Batch Size" in line:
                    config["eff_batch_size"] = line.split(":")[-1].strip()
                elif "Learning Rate" in line:
                    config["lr"] = line.split(":")[-1].strip()
                elif "Max Length" in line:
                    config["max_length"] = line.split(":")[-1].strip()
                elif "BF16 Enabled" in line and "True" in line:
                    config["precision"] = "BF16"
    return config

def parse_progress():
    rows = []
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith('|') and 'Global Step' not in line and '----' not in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 4:
                        rows.append({
                            "step": parts[0],
                            "epoch": parts[1],
                            "train_loss": parts[2],
                            "val_loss": parts[3]
                        })
    return rows

def get_gpu_stats():
    stats = {
        "mem_alloc": "N/A",
        "mem_res": "N/A",
        "utilization": "N/A",
        "temp": "N/A",
        "power": "N/A"
    }
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, text=True
        ).strip().split(',')
        if len(result) >= 5:
            used = float(result[0].strip())
            total = float(result[1].strip())
            stats["mem_alloc"] = f"{used/1024:.2f} GB"
            stats["mem_res"] = f"{total/1024:.2f} GB" 
            stats["utilization"] = f"{result[2].strip()} %"
            stats["temp"] = f"{result[3].strip()} °C"
            stats["power"] = f"{result[4].strip()} W"
    except Exception:
        pass
    return stats

def is_training_running():
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            if p.info['name'] and 'python' in p.info['name'].lower():
                if p.info['cmdline'] and any('train' in arg.lower() for arg in p.info['cmdline']):
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def check_checkpoints():
    global last_known_checkpoints
    current = set()
    if CHECKPOINT_DIR.exists():
        for d in os.listdir(CHECKPOINT_DIR):
            if d.startswith("checkpoint-"):
                current.add(d)
                
    new_checkpoints = current - last_known_checkpoints
    if new_checkpoints:
        for cp in new_checkpoints:
            log_event(f"NEW CHECKPOINT DETECTED: {cp}")
        last_known_checkpoints = current
    
    latest = "N/A"
    if current:
        def extract_num(cp):
            try: return int(cp.split('-')[1])
            except: return 0
        latest = sorted(list(current), key=extract_num)[-1]
        
    return latest

def plot_live_trend(progress):
    try:
        import matplotlib.pyplot as plt
        steps = []
        train_loss = []
        val_loss = []
        for r in progress:
            try:
                s = int(r["step"])
                t = float(r["train_loss"])
                v = float(r["val_loss"])
                steps.append(s)
                train_loss.append(t)
                val_loss.append(v)
            except:
                continue
        
        if len(steps) > 1:
            plt.figure(figsize=(8, 4))
            plt.plot(steps, train_loss, label='Train Loss', color='blue', alpha=0.7)
            plt.plot(steps, val_loss, label='Validation Loss', color='red', alpha=0.9, marker='o')
            plt.title('Live Training Loss Trend')
            plt.xlabel('Global Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            out_file = RUN_DIR / "live_training_curve.png"
            plt.savefig(out_file)
            plt.close()
    except Exception as e:
        pass

def main():
    try:
        while True:
            clear_screen()
            config = parse_config()
            progress = parse_progress()
            gpu_stats = get_gpu_stats()
            is_running = is_training_running()
            latest_cp = check_checkpoints()
            
            # Optional plot
            if len(progress) > 1:
                plot_live_trend(progress)
            
            print("="*60)
            print("       TAMIL INDICBERT MLM — LIVE TRAINING")
            print("="*60)
            print("GPU:\nNVIDIA GeForce RTX 4060 Laptop GPU")
            print("CUDA:\n<detected>")
            print("PyTorch:\n<detected>")
            print("-"*60)
            print("TRAINING")
            print("-"*60)
            
            status = "RUNNING" if is_running else "STOPPED"
            print(f"Status: {status}")
            
            latest_row = progress[-1] if progress else {"step": "0", "epoch": "0.0", "train_loss": "N/A", "val_loss": "N/A"}
            
            print(f"Epoch:\n{latest_row['epoch']} / {total_epochs}")
            print(f"Global Step:\n{latest_row['step']}")
            
            print("-"*60)
            print("LOSS")
            print("-"*60)
            print(f"Current Train Loss:\n{latest_row['train_loss']}")
            print(f"Validation Loss:\n{latest_row['val_loss']}")
            try:
                ppl = math.exp(float(latest_row['val_loss']))
                print(f"Perplexity:\n{ppl:.4f}")
            except:
                print("Perplexity:\nN/A")
                
            print("-"*60)
            print("OPTIMIZATION")
            print("-"*60)
            print(f"Learning Rate:\n{config['lr']}")
            print(f"Batch Size:\n{config['batch_size']}")
            print(f"Gradient Accumulation:\n{config['grad_acc']}")
            print(f"Effective Batch Size:\n{config['eff_batch_size']}")
            print(f"Precision:\n{config['precision']}")
            print(f"Max Length:\n{config['max_length']}")
            
            print("-"*60)
            print("GPU")
            print("-"*60)
            print(f"GPU Memory Allocated:\n{gpu_stats['mem_alloc']}")
            print(f"GPU Memory Reserved:\n{gpu_stats['mem_res']}")
            print(f"GPU Memory Utilization:\n{gpu_stats['utilization']}")
            print(f"Temperature:\n{gpu_stats['temp']}")
            print(f"Power:\n{gpu_stats['power']}")
            
            print("-"*60)
            print("CHECKPOINT")
            print("-"*60)
            print(f"Latest Checkpoint:\n{latest_cp}")
            
            print("-"*60)
            print("HEALTH")
            print("-"*60)
            try:
                t_loss = float(latest_row['train_loss'])
                if math.isnan(t_loss): loss_health = "NaN"
                elif math.isinf(t_loss): loss_health = "Inf"
                else: loss_health = "FINITE"
            except:
                loss_health = "FINITE" if progress else "N/A"
            print(f"Loss:\n{loss_health}")
            print(f"Training Process:\n{status}")
            
            if loss_health in ["NaN", "Inf"]:
                print("="*60)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("CRITICAL TRAINING ERROR DETECTED")
                print(f"LOSS IS {loss_health}")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("="*60)
            
            print("="*60)
            print("LIVE HISTORY")
            print("="*60)
            print(f"{'Step':>8} | {'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10}")
            for row in progress[-10:]:
                print(f"{row['step']:>8} | {row['epoch']:>6} | {row['train_loss']:>10} | {row['val_loss']:>10}")
                
            if not is_running and len(progress) > 0:
                print("="*60)
                print("MLM TRAINING COMPLETED (or STOPPED)")
                print("="*60)
                
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nMLM TRAINING CONTINUES INDEPENDENTLY" if is_training_running() else "\nMLM TRAINING STATUS: STOPPED")
        sys.exit(0)

if __name__ == "__main__":
    if not LOG_FILE.exists():
        log_event("Started Live Monitor")
    main()
