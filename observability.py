import cv2
import mediapipe as mp

class ObservabilityLayer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
    def draw(self, frame_bgr, pose_results, angle, neck_dx, is_slouching, baseline, ml_trained=False, good_count=0, bad_count=0):
        """
        Draws skeletal keypoints, connections, and posture metrics onto the BGR frame.
        """
        if pose_results and pose_results.pose_landmarks:
            connection_color = (0, 0, 255) if is_slouching else (0, 255, 0)
            
            self.mp_drawing.draw_landmarks(
                frame_bgr, 
                pose_results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=connection_color, thickness=2)
            )
        
        # Add text overlay
        color = (0, 0, 255) if is_slouching else (0, 255, 0)
        
        if ml_trained:
            status = "SLOUCHING (ML)!" if is_slouching else "GOOD POSTURE (ML)"
        elif baseline is None:
            status = "PRESS 'c' (Geom) OR 'g'/'s' (ML)!"
            color = (0, 255, 255) # Yellow to indicate action needed
        else:
            status = "SLOUCHING (Geom)!" if is_slouching else "GOOD POSTURE"
        
        cv2.putText(frame_bgr, f"Status: {status}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        if angle is not None and neck_dx is not None:
            cv2.putText(frame_bgr, f"Angle: {angle:.1f} deg", (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Neck dx: {neck_dx:.3f}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        
        # ML Data Collection UI
        if not ml_trained:
            cv2.putText(frame_bgr, f"ML Data: Good={good_count}, Bad={bad_count}", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if good_count > 0 and bad_count > 0:
                cv2.putText(frame_bgr, "Press 't' to Train ML", (20, 155), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        
        return frame_bgr
