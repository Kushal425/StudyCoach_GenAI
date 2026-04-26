import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MLEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.log_reg = LogisticRegression(random_state=42)
        
        self.is_trained = False
        self.accuracy = 0.0

    def generate_synthetic_data(self, n_samples=500):
        """Generates synthetic student performance data."""
        np.random.seed(42)
        
        # Features
        quiz_scores = np.random.uniform(40, 100, n_samples)
        time_spent_hours = np.random.uniform(5, 50, n_samples)
        assignments_completed = np.random.randint(0, 10, n_samples)
        
        # We'll create some realistic correlations
        # Pass (1) or Fail (0) based loosely on scores and completion
        pass_prob = (quiz_scores * 0.5) + (time_spent_hours * 1.5) + (assignments_completed * 5)
        pass_prob = pass_prob / np.max(pass_prob)
        passed = (pass_prob > np.median(pass_prob)).astype(int)
        
        df = pd.DataFrame({
            'student_id': range(1, n_samples + 1),
            'quiz_score': quiz_scores,
            'time_spent_hours': time_spent_hours,
            'assignments_completed': assignments_completed,
            'passed': passed
        })
        
        return df

    def train(self, df):
        """Trains the K-Means and Logistic Regression models."""
        features = ['quiz_score', 'time_spent_hours', 'assignments_completed']
        X = df[features]
        y = df['passed']
        
        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Unsupervised: K-Means Clustering (Learner Categories)
        df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Map clusters to readable labels based on average quiz score
        cluster_means = df.groupby('cluster')['quiz_score'].mean()
        sorted_clusters = cluster_means.sort_values().index
        
        cluster_mapping = {
            sorted_clusters[0]: "At-Risk",
            sorted_clusters[1]: "Average",
            sorted_clusters[2]: "High-Performer"
        }
        df['learner_profile'] = df['cluster'].map(cluster_mapping)
        
        # 2. Supervised: Logistic Regression (Pass/Fail Prediction)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.log_reg.fit(X_train, y_train)
        
        y_pred = self.log_reg.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return df
        
    def predict_student(self, quiz_score, time_spent, assignments):
        """Predicts profile and outcome for a single new student."""
        if not self.is_trained:
            raise ValueError("Models are not trained yet.")
            
        student_data = pd.DataFrame({
            'quiz_score': [quiz_score],
            'time_spent_hours': [time_spent],
            'assignments_completed': [assignments]
        })
        
        X_scaled = self.scaler.transform(student_data)
        
        # Predict outcome
        pass_pred = self.log_reg.predict(X_scaled)[0]
        pass_prob = self.log_reg.predict_proba(X_scaled)[0][1]
        
        # Predict cluster
        cluster = self.kmeans.predict(X_scaled)[0]
        
        # Just simple mapping for the single prediction (assuming same random state)
        profile_mapping = {0: "Average", 1: "High-Performer", 2: "At-Risk"} # Needs robust dynamic mapping in prod, hardcoded for this MVP seed 42 based on training
        
        return {
            "pass_prediction": bool(pass_pred),
            "pass_probability": pass_prob,
            "cluster_id": cluster
        }
