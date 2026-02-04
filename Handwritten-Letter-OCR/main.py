import tkinter as tk
import joblib
import os
from src.gui import OCRGUI
from src.feature_extraction import FeatureExtractor

def load_model(filename):
    """Load a model by filename."""
    path = os.path.join("after.train.model", filename)
    if os.path.exists(path):
        print(f"{filename} loaded successfully!")
        return joblib.load(path)
    print(f"{filename} not found! Train first.")
    return None

def load_feature_extractor():
    """Load the feature extractor; create fresh if missing."""
    path = "after.train.model/feature_extractor.pkl"
    os.makedirs("after.train.model", exist_ok=True)

    if os.path.exists(path):
        try:
            fe = joblib.load(path)
            print("Feature extractor loaded successfully!")
            return fe
        except Exception:
            print("Feature extractor corrupted. Creating new.")

    # Create fresh feature extractor
    fe = FeatureExtractor()
    joblib.dump(fe, path)
    print("New FeatureExtractor saved.")
    return fe

def main():
    print("Starting Handwritten Letter OCR Application...")

    # Load both models
    dt_model = load_model("decision_tree_model.pkl")
    rf_model = load_model("random_forest_model.pkl")

    feature_extractor = load_feature_extractor()

    root = tk.Tk()
    app = OCRGUI(root, dt_model=dt_model, rf_model=rf_model, feature_extractor=feature_extractor)
    root.mainloop()

if __name__ == "__main__":
    main()
