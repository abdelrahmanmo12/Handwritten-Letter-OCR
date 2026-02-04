# src/preprocessing.py
import numpy as np

class Preprocessor:
    """
    Required preprocessing:
    - convert images to 28x28
    - fix EMNIST orientation
    - normalize to [0,1]
    """

    def __init__(self, target_size=(28, 28)):
        self.target_size = target_size

    def preprocess_images(self, images):
        processed = []

        for img in images:
            arr = np.asarray(img).ravel()

            # --- 1. Ensure 28x28 shape ---
            if arr.size != 784:
                side = int(np.sqrt(arr.size))
                img2 = arr.reshape(side, side)
                img2 = self._resize_nn(img2, self.target_size)
            else:
                img2 = arr.reshape(28, 28)

            # --- 2. FIX EMNIST ROTATION ---
            img2 = np.rot90(img2, -1)   # rotate back
            img2 = np.fliplr(img2)      # mirror

            # --- 3. Normalize ---
            img2 = img2.astype(np.float32) / 255.0

            processed.append(img2.flatten())

        return np.array(processed)

    def _resize_nn(self, img, target):
        """nearest neighbor resize (simple & assignment-friendly)"""
        h, w = img.shape
        th, tw = target
        out = np.zeros((th, tw), dtype=np.float32)

        for i in range(th):
            for j in range(tw):
                y = int(i * h / th)
                x = int(j * w / tw)
                out[i, j] = img[y, x]

        return out
