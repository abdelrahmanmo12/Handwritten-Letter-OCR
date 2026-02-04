import unittest
import os
import joblib
import numpy as np
import sys
import json
import time
import matplotlib

# ------------------------------------------------------------------
# Use non-GUI backend to avoid Tkinter errors
# ------------------------------------------------------------------
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

# ------------------------------------------------------------------
# Add project root & src to path
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.feature_extraction import FeatureExtractor
from src.evaluation import ModelEvaluator
from run_training import prepare_data

# ------------------------------------------------------------------
K_FOLDS = 5
# ------------------------------------------------------------------


class TestOCRModels(unittest.TestCase):

    # ==============================================================
    # SETUP
    # ==============================================================
    @classmethod
    def setUpClass(cls):
        print("\n==========================")
        print(" Loading OCR Models ")
        print("==========================")

        # Load trained models
        cls.dt_model = joblib.load("after.train.model/decision_tree_model.pkl")
        cls.rf_model = joblib.load("after.train.model/random_forest_model.pkl")
        cls.feature_extractor = joblib.load("after.train.model/feature_extractor.pkl")

        print(" Models loaded.")

        # Load dataset
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path="data")
        cls.X_test = X_test
        cls.y_test = y_test

        print(f" Test dataset loaded: {cls.X_test.shape[0]} samples")

        # Extract features
        cls.X_test_feat = cls.feature_extractor.extract_features_batch(cls.X_test)
        print(" Features extracted.")

        # Evaluator
        cls.evaluator = ModelEvaluator()

        # Results storage
        cls.results = {
            "Decision_Tree": {},
            "Random_Forest": {}
        }

        cls.results_dir = "results"
        os.makedirs(cls.results_dir, exist_ok=True)

    # ==============================================================
    # SPEED MEASUREMENT
    # ==============================================================
    def measure_speed(self, model, X, loops=300):
        start = time.time()
        for _ in range(loops):
            idx = np.random.randint(0, X.shape[0])
            model.predict(X[idx].reshape(1, -1))
        end = time.time()
        return (end - start) / loops  # seconds/sample

    # ==============================================================
    # FULL EVALUATION
    # ==============================================================
    def full_evaluate(self, model, name):
        print(f"\n===== Evaluating {name} =====")

        preds = model.predict(self.X_test_feat)

        # ---------------- ACCURACY ----------------
        accuracy = float((preds == self.y_test).mean())
        print(f"Accuracy: {accuracy:.4f}")

        # ---------------- SPEED ----------------
        speed = float(self.measure_speed(model, self.X_test_feat))
        print(f"Inference speed: {speed * 1000:.3f} ms/sample")

        # ---------------- PER-CLASS ACCURACY ----------------
        per_class_acc_df = self.evaluator.per_class_accuracy(self.y_test, preds)
        per_class_acc_df.to_csv(
            os.path.join(self.results_dir, f"{name}_per_class_accuracy.csv"),
            index=False
        )

        plt.figure(figsize=(10, 5))
        plt.bar(per_class_acc_df["Class"], per_class_acc_df["Accuracy"])
        plt.title(f"{name} Per-Class Accuracy")
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{name}_per_class_accuracy.png"))
        plt.close()

        # ---------------- CLASSIFICATION REPORT ----------------
        clf_report = self.evaluator.generate_classification_report(self.y_test, preds)
        with open(
            os.path.join(self.results_dir, f"{name}_classification_report.json"),
            "w"
        ) as f:
            json.dump(clf_report, f, indent=2)

        # ---------------- CONFUSION MATRIX ----------------
        fig_cm = self.evaluator.plot_confusion_matrix(
            self.y_test, preds, title=f"{name} Confusion Matrix"
        )
        fig_cm.savefig(os.path.join(self.results_dir, f"{name}_confusion_matrix.png"))
        plt.close(fig_cm)

        # ---------------- FEATURE IMPORTANCE ----------------
        if hasattr(model, "feature_importances_"):
            fig_fi = self.evaluator.plot_feature_importance(
                model.feature_importances_, top_k=20
            )
            fig_fi.savefig(os.path.join(self.results_dir, f"{name}_feature_importance.png"))
            plt.close(fig_fi)

        # Save results
        self.results[name]["accuracy"] = accuracy
        self.results[name]["speed"] = speed

    # ==============================================================
    # K-FOLD CROSS VALIDATION
    # ==============================================================
    def k_fold_validation(self, model, name):
        print(f"\n===== {K_FOLDS}-Fold Cross Validation: {name} =====")

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_test_feat)):
            X_train, X_val = self.X_test_feat[train_idx], self.X_test_feat[val_idx]
            y_train, y_val = self.y_test[train_idx], self.y_test[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = float((preds == y_val).mean())
            fold_accuracies.append(acc)

            print(f" Fold {fold + 1}: accuracy = {acc:.4f}")

        self.results[name]["cv_accuracies"] = fold_accuracies
        self.results[name]["cv_mean"] = float(np.mean(fold_accuracies))
        self.results[name]["cv_std"] = float(np.std(fold_accuracies))

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, K_FOLDS + 1), fold_accuracies, marker="o")
        plt.title(f"{name} {K_FOLDS}-Fold Cross Validation")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{name}_kfold.png"))
        plt.close()

    # ==============================================================
    # TESTS
    # ==============================================================
    def test_decision_tree(self):
        self.full_evaluate(self.dt_model, "Decision_Tree")
        self.k_fold_validation(self.dt_model, "Decision_Tree")

    def test_random_forest(self):
        self.full_evaluate(self.rf_model, "Random_Forest")
        self.k_fold_validation(self.rf_model, "Random_Forest")

    # ==============================================================
    # FINAL COMPARISON + SAVE
    # ==============================================================
    @classmethod
    def tearDownClass(cls):
        print("\nSaving evaluation results...")

        # Save JSON
        with open(os.path.join(cls.results_dir, "results.json"), "w") as f:
            json.dump(cls.results, f, indent=2)

        # -------- ACCURACY vs SPEED COMPARISON --------
        model_names = ["Decision Tree", "Random Forest"]
        accuracies = [
            cls.results["Decision_Tree"]["accuracy"],
            cls.results["Random_Forest"]["accuracy"]
        ]
        speeds = [
            cls.results["Decision_Tree"]["speed"] * 1000,
            cls.results["Random_Forest"]["speed"] * 1000
        ]

        fig = cls.evaluator.plot_accuracy_speed_comparison(
            model_names=model_names,
            accuracies=accuracies,
            speeds=speeds
        )

        fig.savefig(os.path.join(cls.results_dir, "accuracy_speed_comparison.png"))
        plt.close(fig)

        print(" Accuracy & speed comparison plot saved.")
        print(" All results saved successfully.\n")


# ==============================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
