import cv2
import numpy as np


class ImagePreprocessor:

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def denoise(self, image):
        return cv2.fastNlMeansDenoising(image)

    def contrast(self, image):
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        return clahe.apply(image)

    def threshold(self, image):
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    def preprocess(self, image):

        if image is None or image.size == 0:
            raise ValueError("Corrupted image: The image data is empty or None.")

        if len(image.shape) < 2:
            raise ValueError(f"Corrupted image: Invalid dimensions {image.shape}")

        if len(image.shape) == 3:
            gray = self.grayscale(image)
        else:
            gray = image

        denoised = self.denoise(gray)

        enhanced = self.contrast(denoised)

        binary = self.threshold(enhanced)

        return binary
