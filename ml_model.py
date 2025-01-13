from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class DrugResponsePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_features(self, data):
        """Prepare features for model training/prediction"""
        features = ['drug_concentration', 'growth_factor', 'time_period']
        X = data[features]
        if not self.is_fitted:
            X_transformed = self.scaler.fit_transform(X)
            self.is_fitted = True
            return X_transformed
        return self.scaler.transform(X)

    def train(self, data):
        """Train the ML model"""
        X = self.prepare_features(data)
        y = data['effectiveness']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        return {
            'train_score': train_score,
            'test_score': test_score
        }

    def predict(self, features):
        """Make predictions using trained model"""
        X = self.prepare_features(features)
        return self.model.predict(X)