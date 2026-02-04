from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import numpy as np

class OCRModel:
    def __init__(self, model_type='decision_tree'):
        self.model_type = model_type
        self.model = None
        self.feature_importances_ = None
        
    def build_model(self, **params):
        """Build the specified model type"""
        if self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(**params)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**params)
        else:
            raise ValueError("Model type must be 'decision_tree' or 'random_forest'")
    
    def train(self, X_train, y_train):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        return self.model.score(X, y)
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return scores
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid, cv=3):
        """Perform hyperparameter tuning"""
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.feature_importances_ = self.model.feature_importances_