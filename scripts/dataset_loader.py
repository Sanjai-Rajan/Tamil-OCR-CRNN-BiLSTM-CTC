import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import numpy as np

class AspectRatioBatchSampler:
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        print("Pre-calculating aspect ratios for dynamic bucketing...")
        self.aspect_ratios = []
        for d in dataset.data:
            try:
                with Image.open(d["image_path"]) as img:
                    self.aspect_ratios.append(img.size[0] / img.size[1])
            except Exception:
                self.aspect_ratios.append(1.0)
                
        self.sorted_indices = sorted(range(len(self.aspect_ratios)), key=lambda i: self.aspect_ratios[i])
        
    def __iter__(self):
        chunks = [self.sorted_indices[i:i + self.batch_size] for i in range(0, len(self.sorted_indices), self.batch_size)]
        if self.drop_last and len(chunks) > 0 and len(chunks[-1]) < self.batch_size:
            chunks = chunks[:-1]
            
        random.shuffle(chunks)
        for chunk in chunks:
            yield chunk
            
    def __len__(self):
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        else:
            return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


class PacketOCRDataset(Dataset):
    def __init__(self, dataset_root: str, language: str, split: str, tokenizer, transform=None, max_packets=None, target_packet=None, replay_previous=False):
        """
        Loads OCR training data from incrementally imported packets.
        
        Args:
            dataset_root: Root of working dataset (data/tamil_ocr_dataset)
            language: Language prefix (tamil)
            split: Split name (train/validation/test)
            tokenizer: Tokenizer instance
            transform: Torchvision transforms
            max_packets: Optional int to limit packets loaded (for debugging)
            target_packet: If specified (e.g. "packet_001"), trains on just this packet (plus replays if configured)
            replay_previous: If True and target_packet is set, will also load packets before the target.
        """
        self.dataset_root = Path(dataset_root)
        self.language = language
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.data = []
        
        self.imported_dir = self.dataset_root / "imported" / self.language / self.split
        self._load_packets(max_packets, target_packet, replay_previous)
        
    def _load_packets(self, max_packets, target_packet, replay_previous):
        if not self.imported_dir.exists():
            print(f"Warning: No imported packets found at {self.imported_dir}")
            return
            
        packet_dirs = sorted([d for d in self.imported_dir.iterdir() if d.is_dir() and d.name.startswith("packet_")])
        
        if target_packet:
            if replay_previous:
                # Load all packets up to the target packet
                valid_dirs = []
                for p in packet_dirs:
                    valid_dirs.append(p)
                    if p.name == target_packet:
                        break
                packet_dirs = valid_dirs
            else:
                # Only load target packet
                packet_dirs = [p for p in packet_dirs if p.name == target_packet]
        
        if max_packets is not None:
            packet_dirs = packet_dirs[:max_packets]
            
        for packet_dir in packet_dirs:
            manifest_path = packet_dir / "manifest.jsonl"
            if manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            img_path = packet_dir / entry["image"]
                            self.data.append({
                                "image_path": str(img_path),
                                "label": entry["label"]
                            })
                            
        print(f"[{self.split.upper()}] Loaded {len(packet_dirs)} packets containing {len(self.data)} total samples.")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        label = item["label"]
        
        try:
            image = Image.open(img_path).convert('L')
        except Exception:
            image = Image.new('L', (100, 32), color=255)
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = image.to(torch.float32)
            
        target = self.tokenizer.encode(label)
        target_length = len(target)
        
        return image, torch.tensor(target, dtype=torch.long), target_length, img_path

class RandomScalePad:
    def __init__(self, min_scale=0.2, max_scale=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def __call__(self, img):
        # Only randomly scale during training
        scale = random.uniform(self.min_scale, self.max_scale)
        if scale == 1.0:
            return img
            
        w, h = img.size
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Downscale text
        shrunk_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create a white canvas of the ORIGINAL size
        canvas = Image.new('L', (w, h), color=255)
        
        # Paste the shrunk image at a random offset
        offset_x = random.randint(0, w - new_w) if w > new_w else 0
        offset_y = random.randint(0, h - new_h) if h > new_h else 0
        canvas.paste(shrunk_img, (offset_x, offset_y))
        
        return canvas

class ResolutionDegradation:
    def __init__(self, probabilities=None):
        if probabilities is None:
            self.probabilities = {
                1.0: 0.30,   # original
                0.75: 0.20,  # mild
                0.50: 0.20,  # moderate
                0.375: 0.15, # strong
                0.25: 0.10,  # very strong
                0.20: 0.05   # extreme
            }
        self.scales = list(self.probabilities.keys())
        self.probs = list(self.probabilities.values())
        
    def __call__(self, img):
        scale = np.random.choice(self.scales, p=self.probs)
        if scale == 1.0:
            return img
            
        w, h = img.size
        # Add small random padding (0-10% of height) to prevent overfitting to exact 0px border
        pad_pct = random.uniform(0, 0.10)
        pad_px = int(h * pad_pct)
        
        if pad_px > 0:
            canvas = Image.new('L', (w + pad_px*2, h + pad_px*2), color=255)
            canvas.paste(img, (pad_px, pad_px))
            img = canvas
            w, h = img.size
            
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Downscale (simulate low res loss of information)
        down_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        # Upscale back to original dimensions (simulates blurry/thick strokes)
        up_img = down_img.resize((w, h), Image.Resampling.LANCZOS)
        
        return up_img

class AspectRatioPreservingResize:
    def __init__(self, height=32):
        self.height = height
        
    def __call__(self, img):
        w, h = img.size
        new_w = max(4, round(w * self.height / h))
        return img.resize((new_w, self.height), Image.Resampling.LANCZOS)

def crnn_collate_fn(batch):
    if len(batch[0]) == 4:
        images, targets, target_lengths, image_paths = zip(*batch)
    else:
        images, targets, target_lengths = zip(*batch)
        image_paths = None
    
    actual_widths = [img.size(2) for img in images]
    max_width = max(actual_widths)
    height = images[0].size(1)
    
    # Background padding value (white). Since ToTensor converts to 0..1 and 
    # Normalize subtracts 0.5 and divides by 0.5, white (255) becomes 1.0.
    padded_images = torch.ones(len(images), 1, height, max_width) * 1.0
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.size(2)] = img
        
    targets_flat = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    actual_widths = torch.tensor(actual_widths, dtype=torch.long)
    
    if image_paths is not None:
        return padded_images, targets_flat, target_lengths, actual_widths, image_paths
    return padded_images, targets_flat, target_lengths, actual_widths

def get_dataloader(dataset_root, language, split, tokenizer, batch_size=32, num_workers=4, max_packets=None, target_packet=None, replay_previous=False, shuffle=True, use_bucketing=False, use_multiscale_aug=False, use_res_degradation=False):
    
    transform_list = []
    if use_multiscale_aug and split == "train":
        transform_list.append(RandomScalePad(min_scale=0.2, max_scale=1.0))
        
    if use_res_degradation and split == "train":
        import numpy as np
        transform_list.append(ResolutionDegradation())
        
    transform_list.extend([
        AspectRatioPreservingResize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform = transforms.Compose(transform_list)
    
    dataset = PacketOCRDataset(
        dataset_root=dataset_root,
        language=language,
        split=split,
        tokenizer=tokenizer, 
        transform=transform,
        max_packets=max_packets,
        target_packet=target_packet,
        replay_previous=replay_previous
    )
    
    if len(dataset) == 0:
        return None
        
    if use_bucketing:
        batch_sampler = AspectRatioBatchSampler(dataset, batch_size, drop_last=False)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=crnn_collate_fn,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=crnn_collate_fn,
            pin_memory=True
        )
    
    return dataloader
