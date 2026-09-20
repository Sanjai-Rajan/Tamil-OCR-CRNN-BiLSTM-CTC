import argparse
import yaml
import json
from pathlib import Path

def setup_argparse():
    parser = argparse.ArgumentParser(description="Show Dataset Status")
    parser.add_argument("--language", type=str, default="tamil", help="Language prefix for files")
    return parser

def load_config(config_path="config/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    config = load_config()
    
    source_root = Path(config["dataset"]["source_root"])
    working_root = Path(config["dataset"]["working_root"])
    
    print(f"\n{args.language.capitalize()} Dataset Status")
    print("====================")
    
    print("\nOriginal dataset:")
    if source_root.exists():
        print("Available")
    else:
        print("MISSING")
        
    registry_path = working_root / "manifests" / args.language / "packet_registry.json"
    registry = {"train": {}, "validation": {}, "test": {}}
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
    for split in ["train", "validation", "test"]:
        print(f"\n{split.capitalize()}:")
        split_reg = registry.get(split, {})
        if not split_reg:
            print("No packets found in registry")
            continue
            
        for packet_name, info in sorted(split_reg.items()):
            status = info.get("status", "NOT IMPORTED").upper()
            print(f"{packet_name}  {status}")
            
    print("\nTraining:")
    training_state_path = working_root / "reports" / args.language / "training_state.json"
    if training_state_path.exists():
        with open(training_state_path, 'r') as f:
            state = json.load(f)
        print(f"Global epoch: {state.get('global_epoch', 0)}")
        print(f"Last packet: {state.get('current_packet', 'None')}")
        print(f"Checkpoint: {state.get('last_checkpoint', 'None')}")
    else:
        print("Global epoch: 0")
        print("Last packet: None")
        print("Checkpoint: None")

if __name__ == "__main__":
    main()
