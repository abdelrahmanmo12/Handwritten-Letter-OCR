# src/evaluation.py
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

# --- Ensure project root is in sys.path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import FeatureExtractor

class ModelEvaluator:
    def __init__(self, class_names=None):
        if class_names is None:
            # A-Z letters
            self.class_names = [chr(i) for i in range(65, 91)]  # A to Z
        else:
            self.class_names = class_names
        
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        return classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", figsize=(12, 10)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_feature_importance(self, feature_importances, feature_names=None, top_k=20, figsize=(10, 6)):
        """Plot feature importance"""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importances))]
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })
        
        # Sort and select top features
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_k)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_k} Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        return plt.gcf()
    
    def per_class_accuracy(self, y_true, y_pred):
        """Calculate per-class accuracy"""
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        acc_df = pd.DataFrame({
            'Class': self.class_names,
            'Accuracy': per_class_acc
        })
        
        return acc_df
    
    def compare_models(self, models_results, metric='accuracy'):
        """Compare multiple models' performance"""
        comparison_df = pd.DataFrame(models_results)
        return comparison_df
    

    def plot_accuracy_speed_comparison(
       self,
       model_names,
       accuracies,
       speeds,
       figsize=(10, 4)
    ):
       """
       Plot comparison between models in terms of accuracy and inference speed.

       Parameters:
       - model_names: list[str]  → e.g. ["Decision Tree", "Random Forest"]
       - accuracies: list[float] → e.g. [0.67, 0.90]
       - speeds: list[float]     → ms/sample (lower is better)
       """

       fig, axes = plt.subplots(1, 2, figsize=figsize)

      # --- Accuracy plot ---
       axes[0].bar(model_names, accuracies)
       axes[0].set_title("Model Accuracy Comparison")
       axes[0].set_ylabel("Accuracy")
       axes[0].set_ylim(0, 1)
   
       for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha='center')

    # --- Speed plot ---
       axes[1].bar(model_names, speeds)
       axes[1].set_title("Inference Speed Comparison")
       axes[1].set_ylabel("Time (ms / sample)")

       for i, v in enumerate(speeds):
        axes[1].text(i, v, f"{v:.3f}", ha='center', va='bottom')

       plt.suptitle("Decision Tree vs Random Forest Performance")
       plt.tight_layout()
       return fig

    
    def save_results(self, results_dict, filepath):
        """Save evaluation results to file"""
        import json
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results_dict.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)

# --- Optional test run ---
if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
