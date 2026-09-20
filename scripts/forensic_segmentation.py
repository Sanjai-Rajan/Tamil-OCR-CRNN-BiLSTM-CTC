import sys
import os
import json
from PIL import Image
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.cv.segmenter import Segmenter

def main():
    segmenter = Segmenter()
    images = ["sample.jpg", "sample_test.png", "sample_10lines.png"]
    
    results = {}
    
    for img_path in images:
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path)
        lines = segmenter.segment_lines(img)
        
        line_stats = []
        for line_img in lines:
            words = segmenter.segment_words(line_img)
            
            # Since segment_words returns dicts with 'image' and 'confidence',
            # we can only get the size of the cropped words. 
            # We don't have the original bounding box in the returned dict,
            # but we can get word widths and heights.
            
            word_widths = [w['image'].size[0] for w in words]
            word_heights = [w['image'].size[1] for w in words]
            
            line_stats.append({
                "num_words": len(words),
                "word_widths": word_widths,
                "word_heights": word_heights,
                "line_width": line_img.size[0],
                "line_height": line_img.size[1]
            })
            
        results[img_path] = line_stats
        
    with open("docs/robustness_segmentation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
