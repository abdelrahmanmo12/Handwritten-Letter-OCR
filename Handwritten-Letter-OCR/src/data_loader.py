import numpy as np
import struct
import os
from sklearn.model_selection import train_test_split

class EMNISTLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def _load_idx_images(self, filepath):
        with open(filepath, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    def _load_idx_labels(self, filepath):
        with open(filepath, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def load_emnist_letters(self):
        """
        Loads EMNIST Letters dataset (binary IDX files)
        Returns: X, y
        """
        img_train = "emnist-letters-train-images-idx3-ubyte"
        lbl_train = "emnist-letters-train-labels-idx1-ubyte"
        img_test  = "emnist-letters-test-images-idx3-ubyte"
        lbl_test  = "emnist-letters-test-labels-idx1-ubyte"

        print("Loading EMNIST letters IDX files...")

        X_train = self._load_idx_images(os.path.join(self.data_path, img_train))
        y_train = self._load_idx_labels(os.path.join(self.data_path, lbl_train))

        X_test = self._load_idx_images(os.path.join(self.data_path, img_test))
        y_test = self._load_idx_labels(os.path.join(self.data_path, lbl_test))

        # Convert labels 1–26 → 0–25
        y_train = y_train.copy() - 1
        y_test = y_test.copy() - 1

        # Combine train + test into one dataset
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        return X, y

    def train_val_test_split(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split dataset into train / validation / test sets
        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split out the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Calculate validation ratio relative to remaining data
        val_ratio = val_size / (1 - test_size)

        # Split remaining data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
