import os
import json
import torch
import cv2
import numpy as np
from PIL import Image

from app.inference import OCRService

def create_synthetic_line(words, spacing=15):
    # words is a list of PIL Images
    heights = [w.size[1] for w in words]
    widths = [w.size[0] for w in words]
    
    max_h = max(heights)
    total_w = sum(widths) + spacing * (len(words) - 1)
    
    # White background
    line_img = Image.new('RGB', (total_w, max_h), (255, 255, 255))
    
    x_offset = 0
    for w in words:
        # Centered vertically
        y_offset = (max_h - w.size[1]) // 2
        line_img.paste(w, (x_offset, y_offset))
        x_offset += w.size[0] + spacing
        
    return line_img

def main():
    service = OCRService()
    
    # Let's just create a synthetic line from the first few words of the dataset or our samples
    sample1 = Image.open('sample_test.png')
    sample2 = Image.open('sample.jpg') # This is a large word crop
    
    # Resize sample2 to roughly match sample1 height
    h = sample1.size[1]
    new_w = int(sample2.size[0] * h / sample2.size[1])
    sample2_resized = sample2.resize((new_w, h))
    
    synthetic_line = create_synthetic_line([sample1, sample2_resized, sample1], spacing=10)
    synthetic_line.save("synthetic_test_line.png")
    
    print("Running OCR on synthetic line...")
    res = service.predict(synthetic_line)
    
    print(f"Lines detected: {res['lines_detected']}")
    print(f"Words detected: {res['words_detected']}")
    print(f"Expected words: 3")
    print(f"Predicted Text:\n{res['text']}")
    
if __name__ == "__main__":
    main()
