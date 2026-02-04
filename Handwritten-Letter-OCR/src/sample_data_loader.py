import numpy as np
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class SampleDataLoader:
    """Alternative data loader using sklearn digits dataset for testing"""
    
    def __init__(self):
        self.digits = load_digits()
        
    def load_sample_letters(self):
        """Load digits as sample letters (0-9 as A-J)"""
        X = self.digits.images.reshape(-1, 64)  # 8x8 images
        y = self.digits.target
        
        # Resize to 28x28 and convert to letters A-J
        X_resized = []
        for img in self.digits.images:
            # Simple resize simulation
            img_resized = np.kron(img, np.ones((3, 3)))[:28, :28]
            X_resized.append(img_resized.flatten())
            
        X = np.array(X_resized)
        # Convert digits 0-9 to letters A-J
        y_letters = y % 10  # Keep only 10 classes for simplicity
        
        return X, y_letters
    
    def train_val_test_split(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation and test sets"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test