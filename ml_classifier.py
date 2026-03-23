from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import deque
import joblib
import os

MAX_SAMPLES = 500

class PostureClassifier:
    def __init__(self, smoothing_window=15, model_path="posture_model.pkl"):
        # We use a simple KNN because it requires very little data to be highly accurate
        # and trains instantaneously. 3 neighbors prevents single outlier errors.
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.is_trained = False
        self.samples = deque(maxlen=MAX_SAMPLES)
        self.labels = deque(maxlen=MAX_SAMPLES)
        self._good_count = 0
        self._bad_count = 0

        # Ensure path is absolute relative to the script to prevent macOS CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, model_path)

        # Temporal smoothing prevents rapid jittering by requiring sustained classifications
        self.smoothing_window = smoothing_window
        self.history = deque(maxlen=smoothing_window)
        self._history_sum = 0

        # Auto-load existing model if present
        self.load_model()

    def add_sample(self, flat_landmarks, label):
        """
        Label: 0 for Good Posture, 1 for Slouching
        """
        if len(self.samples) == MAX_SAMPLES:
            removed_label = self.labels[0]
            if removed_label == 0:
                self._good_count -= 1
            else:
                self._bad_count -= 1
        self.samples.append(flat_landmarks)
        self.labels.append(label)
        if label == 0:
            self._good_count += 1
        else:
            self._bad_count += 1
        
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
            return False

        X = np.array([flat_landmarks])
        raw_prediction = self.model.predict(X)[0]

        # Update running sum for the moving average (avoids full recompute each frame)
        if len(self.history) == self.smoothing_window:
            self._history_sum -= self.history[0]
        self.history.append(raw_prediction)
        self._history_sum += raw_prediction

        # Only trigger a slouch if the majority of the last N frames were classified as a slouch.
        # This eliminates micro-jitters!
        if len(self.history) == self.smoothing_window:
            return (self._history_sum / self.smoothing_window) > 0.6

        return raw_prediction == 1

    def get_sample_counts(self):
        return self._good_count, self._bad_count

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
                self.samples = deque(data.get('samples', []), maxlen=MAX_SAMPLES)
                self.labels = deque(data.get('labels', []), maxlen=MAX_SAMPLES)
                self._good_count = sum(1 for l in self.labels if l == 0)
                self._bad_count = sum(1 for l in self.labels if l == 1)
                self.is_trained = True
                print(f"[+] Automatically loaded existing ML model from {self.model_path}!")
                return True
            except Exception as e:
                print(f"[-] Failed to load model from {self.model_path}: {e}")
        return False
