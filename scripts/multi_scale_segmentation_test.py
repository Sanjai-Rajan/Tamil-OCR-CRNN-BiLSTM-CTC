import sys
import os
import cv2
import numpy as np
from PIL import Image

def method_A_vertical_projection(binary, min_line_height=10, padding=2):
    # Standard vertical projection used historically
    h, w = binary.shape
    vertical_projection = np.sum(binary, axis=0)
    
    words = []
    in_word = False
    start_x = 0
    
    threshold = 0
    for i, val in enumerate(vertical_projection):
        if not in_word and val > threshold:
            in_word = True
            start_x = i
        elif in_word and val <= threshold:
            in_word = False
            end_x = i
            words.append((max(0, start_x-padding), min(w, end_x+padding)))
            
    if in_word:
        words.append((max(0, start_x-padding), w))
    
    return words

def method_B_connected_components(binary, padding=2):
    # Just raw CC clustering with very basic gap threshold
    h, w = binary.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    components = []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area > max(3, int(h * 0.01)) and ch > max(2, int(h * 0.05)):
            components.append({'x': x, 'right': x + cw})
            
    if not components:
        return []
        
    components.sort(key=lambda c: c['x'])
    merged = []
    curr = components[0].copy()
    for comp in components[1:]:
        if comp['x'] <= curr['right'] + 3:
            curr['right'] = max(curr['right'], comp['right'])
        else:
            merged.append(curr)
            curr = comp.copy()
    merged.append(curr)
    
    gaps = [merged[i+1]['x'] - merged[i]['right'] for i in range(len(merged)-1)]
    valid_gaps = [g for g in gaps if g > 0]
    
    if len(valid_gaps) > 2:
        threshold = max(4, np.median(valid_gaps) * 1.5)
    else:
        threshold = max(4, int(h * 0.12))
        
    words = []
    curr_word = [merged[0]]
    for i in range(len(merged)-1):
        gap = merged[i+1]['x'] - merged[i]['right']
        if gap > threshold:
            words.append(curr_word)
            curr_word = [merged[i+1]]
        else:
            curr_word.append(merged[i+1])
    words.append(curr_word)
    
    return [(min(c['x'] for c in w_c), max(c['right'] for c in w_c)) for w_c in words]

def method_C_combined(binary, padding=2):
    # Combine projection and CC
    h, w = binary.shape
    
    # Run morphological closing to bridge small gaps WITHIN characters/words first
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(h*0.05) or 1, int(h*0.1) or 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    components = []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        components.append({'x': x, 'right': x + cw})
        
    if not components:
        return []
        
    components.sort(key=lambda c: c['x'])
    
    merged = []
    curr = components[0].copy()
    for comp in components[1:]:
        if comp['x'] <= curr['right']:
            curr['right'] = max(curr['right'], comp['right'])
        else:
            merged.append(curr)
            curr = comp.copy()
    merged.append(curr)
    
    # Measure gaps
    gaps = [merged[i+1]['x'] - merged[i]['right'] for i in range(len(merged)-1)]
    valid_gaps = [g for g in gaps if g > 0]
    
    if len(valid_gaps) > 1:
        threshold = max(max(4, int(h * 0.1)), np.median(valid_gaps) * 1.5)
    else:
        threshold = max(4, int(h * 0.15))
        
    words = []
    curr_word = [merged[0]]
    for i in range(len(merged)-1):
        gap = merged[i+1]['x'] - merged[i]['right']
        if gap > threshold:
            words.append(curr_word)
            curr_word = [merged[i+1]]
        else:
            curr_word.append(merged[i+1])
    words.append(curr_word)
    
    return [(min(c['x'] for c in w_c), max(c['right'] for c in w_c)) for w_c in words]

def evaluate_methods(image_path):
    print(f"\n--- Evaluating {image_path} ---")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found.")
        return
        
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
        
    # We will assume this is a LINE image for the test
    # (or we can just run line segmentation first and take the first line)
    horizontal_projection = np.sum(binary, axis=1)
    threshold = 255 * 3
    in_line = False
    lines = []
    start_y = 0
    for i, val in enumerate(horizontal_projection):
        if not in_line and val > threshold:
            in_line = True
            start_y = i
        elif in_line and val <= threshold:
            in_line = False
            end_y = i
            if (end_y - start_y) >= 10:
                lines.append((start_y, end_y))
    if in_line and (len(horizontal_projection) - start_y) >= 10:
        lines.append((start_y, len(horizontal_projection)))
        
    for idx, (sy, ey) in enumerate(lines):
        line_bin = binary[sy:ey, :]
        print(f"\nLine {idx+1} size: {line_bin.shape}")
        
        words_A = method_A_vertical_projection(line_bin)
        print(f"Method A (Vertical Projection): {len(words_A)} words")
        
        words_B = method_B_connected_components(line_bin)
        print(f"Method B (CC): {len(words_B)} words")
        
        words_C = method_C_combined(line_bin)
        print(f"Method C (Combined Morphology+CC): {len(words_C)} words")

if __name__ == "__main__":
    evaluate_methods("sample_10lines.png")
