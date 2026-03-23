import math
import numpy as np
import mediapipe as mp

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points A, B, and C with B being the vertex.
        Returns the angle in degrees.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def analyze(self, image_rgb):
        """
        Processes an RGB image to find pose landmarks and calculates the posture metrics.
        
        Returns:
            results: The raw MediaPipe pose results
            posture_angle (float): The calculated core angle or None if not detected
            neck_dx (float): The horizontal X-distance between the ear and shoulder or None
        """
        results = self.pose.process(image_rgb)
        posture_angle = None
        neck_dx = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Since the camera is on the left facing in profile, the left side is what we see.
            l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, 
                     landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
            l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                          landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            # Since the hip might not be in frame, we create a pseudo-landmark 
            # directly below the shoulder to represent a vertical torso line.
            vertical_ref = [l_shoulder[0], l_shoulder[1] + 1.0]
            
            posture_angle = self.calculate_angle(l_ear, l_shoulder, vertical_ref)
            
            # Neck horizontal displacement (Text Neck tracking)
            # The difference in X between the ear and shoulder.
            neck_dx = l_ear[0] - l_shoulder[0]
            
        return results, posture_angle, neck_dx

    def get_flat_landmarks(self, results):
        """
        Extracts all 33 3D landmarks into a single flat array for Machine Learning.
        Returns a list of 99 features (x, y, z for each point).
        """
        if not results or not results.pose_landmarks:
            return None
            
        flat = []
        for lm in results.pose_landmarks.landmark:
            flat.extend([lm.x, lm.y, lm.z])
        return flat
