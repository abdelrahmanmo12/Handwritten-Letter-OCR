# run_training.py  — FINAL FIXED VERSION
import os
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.feature_extraction import FeatureExtractor


# -------------------------------------------------------------------------------------
# FALLBACK EMNIST LOADER
# -------------------------------------------------------------------------------------
class EMNISTLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def _load_idx_images(self, filepath):
        import struct
        with open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    def _load_idx_labels(self, filepath):
        import struct
        with open(filepath, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def load_emnist_letters(self):
        img_train = "emnist-letters-train-images-idx3-ubyte"
        lbl_train = "emnist-letters-train-labels-idx1-ubyte"
        img_test  = "emnist-letters-test-images-idx3-ubyte"
        lbl_test  = "emnist-letters-test-labels-idx1-ubyte"

        X_train = self._load_idx_images(os.path.join(self.data_path, img_train))
        y_train = self._load_idx_labels(os.path.join(self.data_path, lbl_train))
        X_test  = self._load_idx_images(os.path.join(self.data_path, img_test))
        y_test  = self._load_idx_labels(os.path.join(self.data_path, lbl_test))

        y_train = y_train.copy() - 1
        y_test  = y_test.copy() - 1

        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        return X, y

    def train_val_test_split(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp)

        return X_train, X_val, X_test, y_train, y_val, y_test



# -------------------------------------------------------------------------------------
# PREPROCESSING (normalise)
# -------------------------------------------------------------------------------------
def normalize(x):
    x = x.astype(np.float32)
    x = x / 255.0
    return x


# -------------------------------------------------------------------------------------
# PREPARE DATA
# -------------------------------------------------------------------------------------
def prepare_data(data_path):
    loader = EMNISTLoader(data_path)

    print("Loading EMNIST letters IDX files...")
    X, y = loader.load_emnist_letters()
    print(f"Full dataset shape: {X.shape}, Labels: {y.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = loader.train_val_test_split(X, y)

    print("Preprocessing images to 28x28 and normalizing...")
    X_train = normalize(X_train)
    X_val   = normalize(X_val)
    X_test  = normalize(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test



# -------------------------------------------------------------------------------------
# APPLY FEATURE EXTRACTION (HOG + PCA)
# -------------------------------------------------------------------------------------
def apply_feature_extraction(X_train, X_val, X_test):
    fe = FeatureExtractor(
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        n_components=150
    )

    print("Fitting HOG + PCA on training data...")
    X_train_feat = fe.fit_transform(X_train)

    print("Transforming validation and test data...")
    X_val_feat = fe.transform(X_val)
    X_test_feat = fe.transform(X_test)

    os.makedirs("after.train.model", exist_ok=True)
    joblib.dump(fe, "after.train.model/feature_extractor.pkl")
    print("Feature extractor saved to after.train.model/feature_extractor.pkl")

    return X_train_feat, X_val_feat, X_test_feat




# -------------------------------------------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------------------------------------------
def train_and_save_models(X_train, y_train, X_test, y_test):

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    print("Training Decision Tree...")
    dt.fit(X_train, y_train)

    joblib.dump(dt, "after.train.model/decision_tree_model.pkl")
    print("Saved Decision Tree model")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    joblib.dump(rf, "after.train.model/random_forest_model.pkl")
    print("Saved Random Forest model")



# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------
def main(data_path=r"C:\Users\Abd elrahman\AppData\Local\emnist"):

    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path)

    print("Extracting HOG + PCA features...")
    X_train_f, X_val_f, X_test_f = apply_feature_extraction(X_train, X_val, X_test)


    print("Training models...")
    train_and_save_models(X_train_f, y_train, X_test_f, y_test)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
