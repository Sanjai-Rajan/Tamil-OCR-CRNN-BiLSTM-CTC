import cv2
import numpy as np
from PIL import Image

class Segmenter:
    def __init__(self, min_line_height=5, min_line_gap=8, padding=6):
        self.min_line_height = min_line_height
        self.min_line_gap = min_line_gap
        self.padding = padding

    def segment_lines(self, image):
        """
        Takes a PIL Image or numpy array.
        Returns a list of PIL Image crops (top-to-bottom order).
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert('RGB'))
            gray = np.array(image.convert('L'))
        else:
            img_np = image.copy()
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_np.copy()

        # Robust binarization (Otsu's thresholding)
        # Invert so text is white (255) and background is black (0)
        # Sometimes images have dark background, but Otsu handles assuming bimodal.
        # Let's check mean intensity to ensure white text on black background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # If the original image was mostly dark, binary_inv might invert it wrong.
        # Safe assumption: document has more background (white) than text (black).
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # Calculate horizontal projection profile
        horizontal_projection = np.sum(binary, axis=1)

        # Detect text-line bands
        lines = []
        in_line = False
        start_y = 0

        # A small threshold instead of strictly > 0 to handle minor noise
        threshold = 255 * 3  # At least 3 white pixels

        for i, val in enumerate(horizontal_projection):
            if not in_line and val > threshold:
                in_line = True
                start_y = i
            elif in_line and val <= threshold:
                in_line = False
                end_y = i
                
                if (end_y - start_y) >= self.min_line_height:
                    lines.append((start_y, end_y))
        
        # If still in line at the end
        if in_line:
            end_y = len(horizontal_projection)
            if (end_y - start_y) >= self.min_line_height:
                lines.append((start_y, end_y))

        # Merge fragmented bands
        merged_lines = []
        if lines:
            curr_start, curr_end = lines[0]
            for i in range(1, len(lines)):
                next_start, next_end = lines[i]
                if next_start - curr_end <= self.min_line_gap:
                    # Merge
                    curr_end = next_end
                else:
                    merged_lines.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_lines.append((curr_start, curr_end))

        # Crop lines with padding
        segments = []
        h, w = gray.shape
        for start_y, end_y in merged_lines:
            y1 = max(0, start_y - self.padding)
            y2 = min(h, end_y + self.padding)
            
            # Find the horizontal bounds of the text in this line to tightly crop width
            line_binary = binary[start_y:end_y, :]
            vertical_projection = np.sum(line_binary, axis=0)
            non_zero_x = np.where(vertical_projection > 0)[0]
            
            if len(non_zero_x) > 0:
                x1 = max(0, non_zero_x[0] - self.padding)
                x2 = min(w, non_zero_x[-1] + self.padding)
            else:
                x1, x2 = 0, w
            
            if isinstance(image, Image.Image):
                roi = image.crop((x1, y1, x2, y2))
            else:
                roi = img_np[y1:y2, x1:x2]
            segments.append(roi)

        # If it failed to segment anything (e.g. noise), return original
        if not segments:
            return [image]

        return segments

    def segment_words(self, line_image):
        """
        Takes a PIL Image or numpy array of a single text line.
        Returns a list of dicts: {'image': PIL Image / numpy array, 'confidence': float}
        """
        if isinstance(line_image, Image.Image):
            img_np = np.array(line_image.convert('RGB'))
            gray = np.array(line_image.convert('L'))
        else:
            img_np = line_image.copy()
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_np.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        h, w = binary.shape
        
        # 1. Morphological closing to bridge small gaps WITHIN characters/words
        # Tamil characters often consist of multiple strokes or diacritics
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(h * 0.05)), max(2, int(h * 0.1))))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        components = []
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            # Filter noise
            if area > max(3, int(h * 0.01)) and ch > max(2, int(h * 0.05)):
                components.append({'x': x, 'y': y, 'w': cw, 'h': ch, 'right': x + cw})
                
        if not components:
            return [{'image': line_image, 'confidence': 0.0}]
            
        # Sort left-to-right based on x
        components.sort(key=lambda c: c['x'])
        
        # 2. Merge overlapping components
        merged = []
        curr = components[0].copy()
        for comp in components[1:]:
            # If overlapping or touching
            if comp['x'] <= curr['right']:
                curr['right'] = max(curr['right'], comp['right'])
                curr['y'] = min(curr['y'], comp['y'])
                curr['h'] = max(curr['y'] + curr['h'], comp['y'] + comp['h']) - curr['y']
                curr['x'] = min(curr['x'], comp['x'])
                curr['w'] = curr['right'] - curr['x']
            else:
                merged.append(curr)
                curr = comp.copy()
        merged.append(curr)

        if len(merged) <= 1:
            # Single giant component (e.g. physically touching words, zero-gap)
            aspect_ratio = w / max(1, h)
            conf = 1.0 if aspect_ratio < 4.0 else 0.5
            return [{'image': line_image, 'confidence': conf}]

        # 3. Extract gaps
        gaps = [merged[i+1]['x'] - merged[i]['right'] for i in range(len(merged)-1)]
        valid_gaps = [g for g in gaps if g > 0]
        
        # 4. Adaptive threshold
        if len(valid_gaps) > 1:
            median_gap = np.median(valid_gaps)
            # Threshold: slightly larger than median component gap, but at least 10% of height
            threshold = max(max(4, int(h * 0.1)), median_gap * 1.5)
        else:
            threshold = max(4, int(h * 0.15))
            
        # 5. Split into words based on threshold
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

        # 6. Crop the words
        segments = []
        for word_comps in words:
            x1 = min(c['x'] for c in word_comps)
            x2 = max(c['right'] for c in word_comps)
            
            aspect_ratio = (x2 - x1) / max(1, h)
            conf = 1.0 if aspect_ratio < 4.0 else 0.5
            
            x1 = max(0, x1 - self.padding)
            x2 = min(w, x2 + self.padding)
            
            # Vertical tight crop
            word_bin = binary[:, x1:x2]
            hp = np.sum(word_bin, axis=1)
            y_nz = np.where(hp > 0)[0]
            if len(y_nz) > 0:
                y1 = max(0, y_nz[0] - self.padding)
                y2 = min(h, y_nz[-1] + self.padding)
            else:
                y1, y2 = 0, h
                
            if isinstance(line_image, Image.Image):
                roi = line_image.crop((x1, y1, x2, y2))
            else:
                roi = img_np[y1:y2, x1:x2]
            segments.append({'image': roi, 'confidence': conf})
            
        return segments
