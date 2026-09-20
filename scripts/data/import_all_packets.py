import subprocess
import math
import sys
from pathlib import Path

def run_import(split, packet_idx):
    print(f"Starting import for {split} packet {packet_idx}...")
    cmd = [
        sys.executable, 
        "scripts/import_tamil_packet.py", 
        "--language", "tamil", 
        "--split", split, 
        "--packet", str(packet_idx)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error importing {split} packet {packet_idx}:")
        print(result.stderr)
        return False
    print(f"Successfully imported {split} packet {packet_idx}.")
    return True

def main():
    packet_size = 5000
    splits = {
        "train": 75736,
        "validation": 11598,
        "test": 16184
    }
    
    total_failures = 0
    
    for split, total_images in splits.items():
        num_packets = math.ceil(total_images / packet_size)
        print(f"\n--- Importing {split.upper()} packets (Total: {num_packets}) ---")
        
        for p in range(1, num_packets + 1):
            success = run_import(split, p)
            if not success:
                total_failures += 1
                print("Stopping due to failure.")
                return

    if total_failures == 0:
        print("\nAll packets imported successfully!")
        
        # Run dataset status to confirm
        print("\nFinal Dataset Status:")
        subprocess.run([sys.executable, "scripts/dataset_status.py", "--language", "tamil"])
    else:
        print(f"\nImport finished with {total_failures} failures.")

if __name__ == "__main__":
    main()
