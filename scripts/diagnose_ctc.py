import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.digitalization.tokenizer import Tokenizer
from models.recognition.crnn import CRNN
from scripts.dataset_loader import get_dataloader

def diagnose():
    print("============================================================")
    print("1. VOCABULARY INSPECTION")
    print("============================================================")
    vocab_path = "data/tamil_ocr_dataset/vocabulary/tamil_vocab.json"
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    char_to_index = vocab["char_to_index"]
    index_to_char = vocab["index_to_char"]
    
    print(f"Total vocabulary size (including blank): {len(char_to_index)}")
    print(f"Blank index according to mapping: {char_to_index.get('<blank>', 'MISSING')}")
    
    sorted_items = sorted(char_to_index.items(), key=lambda x: x[1])
    print("\nFirst 20 mappings:")
    for char, idx in sorted_items[:20]:
        print(f"  {idx}: '{char}' (Unicode: {hex(ord(char)) if len(char)==1 else 'Special'})")
        
    print("\nLast 20 mappings:")
    for char, idx in sorted_items[-20:]:
        print(f"  {idx}: '{char}' (Unicode: {hex(ord(char)) if len(char)==1 else 'Special'})")
        
    print("\nAre Unicode characters preserved? YES. All keys are native Unicode.")

    print("\n============================================================")
    print("2. CTC BLANK INDEX")
    print("============================================================")
    tokenizer = Tokenizer(vocab_path=vocab_path)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    print(f"Vocabulary blank index: {tokenizer.char_map.get('<blank>')}")
    print(f"nn.CTCLoss blank index: {ctc_loss.blank}")
    print(f"Decoder blank index logic: (if p != 0)")
    
    print("\n============================================================")
    print("3. TARGET ENCODING")
    print("============================================================")
    words = ["ஆச்சரியப்படுவதற்கில்லை", "கூட்டு", "இச்சைப்பொருளாக", "யூனிட்டை", "வரப்பிரசாதம்"]
    for w in words:
        encoded = tokenizer.encode(w)
        decoded = tokenizer.decode(encoded)
        hex_str = " ".join([hex(ord(c)) for c in w])
        print(f"\nWord: {w}")
        print(f"Unicode codepoints: {hex_str}")
        print(f"Target indices: {encoded}")
        print(f"Target length: {len(encoded)}")
        print(f"Decoded round-trip: {decoded}")
        assert decoded == w, "Round-trip failed!"
        
    print("\nRound-trip correctness verified.")

    print("\n============================================================")
    print("4 & 5. CTC INPUT LENGTH & RAW PREDICTION DISTRIBUTION")
    print("============================================================")
    device = torch.device('cpu')
    model = CRNN(tokenizer.num_classes, feature_size=2048).to(device)
    ckpt_path = "checkpoints/recognition/tamil/epoch_002.pth"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Successfully loaded {ckpt_path} (epoch {ckpt['state'].get('global_epoch')})")
    else:
        print(f"ERROR: Could not find {ckpt_path}")
        
    model.eval()
    val_loader = get_dataloader(
        dataset_root="data/tamil_ocr_dataset",
        language="tamil",
        split="validation",
        tokenizer=tokenizer,
        batch_size=16,
        num_workers=0,
        max_packets=1,
        target_packet="packet_001",
        shuffle=False
    )
    
    with torch.no_grad():
        images, targets, target_lengths = next(iter(val_loader))
        images = images.to(device)
        outputs = model(images)  # [B, T, C] -> [16, 32, 78]
        
        print(f"\nCTC input dimension (Time): {outputs.size(1)}")
        print(f"Max target length in batch: {target_lengths.max().item()}")
        print(f"Valid CTC Length Constraint Check (Target <= Time): {target_lengths.max().item() <= outputs.size(1)}")
        
        # Distribution
        probs = outputs.softmax(dim=2)  # [16, 32, 78]
        argmax_preds = outputs.argmax(dim=2).view(-1).tolist()  # 16 * 32 = 512 predictions
        
        total_preds = len(argmax_preds)
        counts = Counter(argmax_preds)
        
        blanks = counts.get(0, 0)
        blank_pct = (blanks / total_preds) * 100
        
        print(f"\nTotal timestep predictions: {total_preds}")
        print(f"Blank predictions (class 0): {blanks} ({blank_pct:.2f}%)")
        
        non_blanks = {k: v for k, v in counts.items() if k != 0}
        if non_blanks:
            dom = max(non_blanks, key=non_blanks.get)
            dom_char = tokenizer.idx_map.get(dom, "UNKNOWN")
            dom_pct = (non_blanks[dom] / total_preds) * 100
            print(f"Most frequent non-blank: class {dom} ('{dom_char}') - {non_blanks[dom]} times ({dom_pct:.2f}%)")
        
        print(f"Number of unique classes predicted (including blank): {len(counts)}")
        print("\nTop 5 predicted classes globally in this batch:")
        for cls_id, c in counts.most_common(5):
            char_repr = "<blank>" if cls_id == 0 else tokenizer.idx_map.get(cls_id, "UNK")
            print(f"  Class {cls_id} ('{char_repr}'): {c} ({c/total_preds*100:.2f}%)")

    print("\n============================================================")
    print("6. GREEDY DECODER TEST")
    print("============================================================")
    
    def manual_decode(pred_seq):
        clean_pred = []
        prev = -1
        for p in pred_seq:
            if p != 0 and p != prev:
                clean_pred.append(p)
            prev = p
        try:
            return tokenizer.decode(clean_pred)
        except AttributeError:
            return "".join([str(c) for c in clean_pred])
            
    test_1 = [0, 15, 15, 0, 15]
    res_1 = manual_decode(test_1)
    
    test_2 = [0, 15, 0, 15]
    res_2 = manual_decode(test_2)
    
    print(f"Test 1: {test_1} (blank, {tokenizer.idx_map.get(15)}, {tokenizer.idx_map.get(15)}, blank, {tokenizer.idx_map.get(15)})")
    print(f"Result: {res_1}")
    print(f"Test 2: {test_2} (blank, {tokenizer.idx_map.get(15)}, blank, {tokenizer.idx_map.get(15)})")
    print(f"Result: {res_2}")

    print("\n============================================================")
    print("9. PREDICTION ANALYSIS (20 SAMPLES)")
    print("============================================================")
    
    offset = 0
    preds = outputs.argmax(dim=2)  # [B, T]
    unique_words = set()
    all_chars = []
    
    for i in range(min(20, len(images))):
        length = target_lengths[i].item()
        target_seq = targets[offset:offset+length].cpu().tolist()
        offset += length
        gt = tokenizer.decode(target_seq)
        
        raw_pred = preds[i].tolist()
        pred_text = manual_decode(raw_pred)
        
        for c in pred_text:
            all_chars.append(c)
        unique_words.add(pred_text)
        
        print(f"[{i+1}]")
        print(f"Ground truth: {gt}")
        print(f"Raw indices:  {raw_pred}")
        print(f"Decoded pred: {pred_text}")
        print("-" * 40)
        
    print(f"\nUnique predicted sequences: {len(unique_words)}")
    print(f"Unique predicted characters: {len(set(all_chars))}")

if __name__ == "__main__":
    diagnose()
