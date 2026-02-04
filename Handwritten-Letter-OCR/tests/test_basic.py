import sys
import os
import joblib
import numpy as np

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from run_training import prepare_data
from feature_extraction import FeatureExtractor
from evaluation import ModelEvaluator

def main():
    # Paths
    data_path = r"C:\Users\Abd elrahman\AppData\Local\emnist"  # Update if needed
    model_path = os.path.join('..', 'models', 'random_forest_model.pkl')
    fe_path = os.path.join('..', 'models', 'feature_extractor.pkl')

    # Load trained model
    if not os.path.exists(model_path):
        print("Random Forest model not found. Train the model first.")
        return
    model = joblib.load(model_path)
    print("Random Forest model loaded.")

    # Load feature extractor
    if not os.path.exists(fe_path):
        print("Feature extractor not found. Creating a fresh one (untrained).")
        fe = FeatureExtractor()
    else:
        fe = joblib.load(fe_path)
        print("Feature extractor loaded.")

    # Prepare EMNIST test data
    _, _, X_test, _, _, y_test = prepare_data(data_path)
    print(f"Loaded test data: {X_test.shape}, Labels: {y_test.shape}")

    # Extract features for test set
    X_test_feat = fe.extract_features_batch(X_test)

    # Predict
    y_pred = model.predict(X_test_feat)

    # Evaluate
    evaluator = ModelEvaluator()
    report = evaluator.generate_classification_report(y_test, y_pred)
    print("\nClassification Report:")
    for cls, metrics in report.items():
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{cls}: {metrics}")

    acc_df = evaluator.per_class_accuracy(y_test, y_pred)
    print("\nPer-Class Accuracy:")
    print(acc_df)

    # Plot confusion matrix
    cm_fig = evaluator.plot_confusion_matrix(y_test, y_pred)
    cm_fig.savefig("results/confusion_matrix.png")
    print("\nConfusion matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    main()
