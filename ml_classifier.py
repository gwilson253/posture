from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import deque
import joblib
import os

class PostureClassifier:
    def __init__(self, smoothing_window=15, model_path="posture_model.pkl"):
        # We use a simple KNN because it requires very little data to be highly accurate
        # and trains instantaneously. 3 neighbors prevents single outlier errors.
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        self.samples = []
        self.labels = []
        
        # Ensure path is absolute relative to the script to prevent macOS CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, model_path)
        
        # Temporal smoothing prevents rapid jittering by requiring sustained classifications
        self.smoothing_window = smoothing_window
        self.history = deque(maxlen=smoothing_window)
        
        # Auto-load existing model if present
        self.load_model()
        
    def add_sample(self, flat_landmarks, label):
        """
        Label: 0 for Good Posture, 1 for Slouching
        """
        self.samples.append(flat_landmarks)
        self.labels.append(label)
        
    def train(self):
        if len(set(self.labels)) < 2:
            print("Warning: Model needs at least one 'Good' and one 'Slouch' sample to train.")
            return False
            
        print("Training KNN classifier on current dataset...")
        X = np.array(self.samples)
        y = np.array(self.labels)
        
        # If we have very few samples, reduce n_neighbors temporarily
        if len(self.samples) < 3:
            self.model.n_neighbors = 1
        else:
            self.model.n_neighbors = 3
            
        self.model.fit(X, y)
        self.is_trained = True
        
        print("Training complete!")
        return True
        
    def predict(self, flat_landmarks):
        """
        Returns True if slouching is detected and sustained over the smoothing window.
        """
        if not self.is_trained:
            # If not trained, fallback to not slouching
            return False
            
        X = np.array([flat_landmarks])
        raw_prediction = self.model.predict(X)[0]
        
        # Signal Smoothing (Moving Average)
        self.history.append(raw_prediction)
        
        # Only trigger a slouch if the majority of the last N frames were classified as a slouch.
        # This eliminates micro-jitters!
        if len(self.history) == self.smoothing_window:
            slouch_ratio = sum(self.history) / self.smoothing_window
            # If > 60% of recent frames are slouches, confirm it.
            return slouch_ratio > 0.6
            
        return raw_prediction == 1
        
    def get_sample_counts(self):
        good = sum(1 for label in self.labels if label == 0)
        bad = sum(1 for label in self.labels if label == 1)
        return good, bad

    def save_model(self):
        """Saves the trained model and dataset to disk."""
        if self.is_trained:
            data = {
                'model': self.model,
                'samples': self.samples,
                'labels': self.labels
            }
            joblib.dump(data, self.model_path)
            print(f"\n[+] Model and {len(self.samples)} samples saved to {self.model_path}!")
            return True
        return False
        
    def load_model(self):
        """Attempts to load a previously saved model on startup."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.samples = data.get('samples', [])
                self.labels = data.get('labels', [])
                self.is_trained = True
                print(f"[+] Automatically loaded existing ML model from {self.model_path}!")
                return True
            except Exception as e:
                print(f"[-] Failed to load model from {self.model_path}: {e}")
        return False
