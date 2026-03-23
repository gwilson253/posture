import os
os.nice(10)  # Lower process priority so it yields to foreground apps
# Cap thread pools before importing OpenCV or MediaPipe (which spin up threads on import)
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
import cv2
cv2.setNumThreads(2)
from posture_analyzer import PostureAnalyzer
from observability import ObservabilityLayer
from alert_manager import AlertManager
from ml_classifier import PostureClassifier

def main():
    print("Initializing components...")
    analyzer = PostureAnalyzer()
    observability = ObservabilityLayer()
    alert_manager = AlertManager(slouch_duration_threshold=3.0, cooldown_duration=10.0)
    classifier = PostureClassifier(smoothing_window=15)
    
    # Initialize the camera (0 is usually the default webcam or USB camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    # Posture heuristics:
    # Instead of a hardcoded angle, we now use a calibration baseline!
    baseline_angle = None
    baseline_neck_dx = None
    
    # Deviations that trigger a slouch
    ANGLE_DROP_THRESHOLD = 15.0      # degrees dropped from baseline
    NECK_DX_THRESHOLD = 0.04         # percentage of frame width the neck moved forward

    print("Starting Posture Monitor...")
    print("Press 'c' to CALIBRATE geometry baseline.")
    print("Press 'g' to RECORD Good Posture for ML (Hold or Tap).")
    print("Press 's' to RECORD Slouching for ML (Hold or Tap).")
    print("Press 't' to TRAIN the ML model.")
    print("Press 'w' to SAVE the trained ML model permanently.")
    print("Press 'q' in the video window to quit.")

    frame_count = 0
    results, angle, neck_dx, flat_landmarks = None, None, None, None

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab camera frame. Exiting...")
            break

        frame_count += 1

        # Run MediaPipe inference every other frame — posture changes slowly
        if frame_count % 2 == 0:
            # Downscale before inference — MediaPipe doesn't need full camera resolution
            small_frame = cv2.resize(frame, (480, 270))
            image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Performance optimization: prevent modifying image to save memory during processing
            image_rgb.flags.writeable = False

            # Analyze posture
            results, angle, neck_dx = analyzer.analyze(image_rgb)
            flat_landmarks = analyzer.get_flat_landmarks(results)
        
        # Check keystrokes
        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and angle is not None and neck_dx is not None:
            baseline_angle = angle
            baseline_neck_dx = neck_dx
            print(f"Calibration Set! Baseline Angle: {baseline_angle:.1f}, Neck DX: {baseline_neck_dx:.3f}")
        elif key == ord('g') and flat_landmarks is not None:
            classifier.add_sample(flat_landmarks, 0)
        elif key == ord('s') and flat_landmarks is not None:
            classifier.add_sample(flat_landmarks, 1)
        elif key == ord('t'):
            classifier.train()
        elif key == ord('w'):
            classifier.save_model()
        
        # Update alerting state
        is_slouching = False
        
        # ML classification takes priority if trained
        if classifier.is_trained and flat_landmarks is not None:
            is_slouching = classifier.predict(flat_landmarks)
            alert_manager.run_check(is_slouching)
        elif baseline_angle is not None and baseline_neck_dx is not None and angle is not None and neck_dx is not None:
            # Fallback to pure geometry
            angle_slouch = (baseline_angle - angle) > ANGLE_DROP_THRESHOLD
            neck_slouch = abs(neck_dx - baseline_neck_dx) > NECK_DX_THRESHOLD
            
            is_slouching = angle_slouch or neck_slouch
            alert_manager.run_check(is_slouching)
            
        # Draw the observability overlay
        good_count, bad_count = classifier.get_sample_counts()
        annotated_frame = observability.draw(
            frame, results, angle, neck_dx, is_slouching, baseline_angle,
            ml_trained=classifier.is_trained, good_count=good_count, bad_count=bad_count
        )
        
        # Display the result
        cv2.imshow('Posture Monitor Observability Layer', annotated_frame)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
