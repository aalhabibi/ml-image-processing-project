#!/usr/bin/env python3
"""
Real-Time Waste Classification System
Uses webcam to classify waste materials in real-time
Press 'q' to quit, 's' to save screenshot
"""

import pickle
import cv2
import numpy as np
import sys
from pathlib import Path
from collections import deque
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.feature_extraction import FeatureExtractor


class RealtimeWasteClassifier:
    def __init__(
        self,
        model_path="best_model/best_model.pkl",
        scaler_path="features/scaler.pkl",
        *,
        overlay_alpha=0.65,
    ):
        """Initialize the real-time classifier"""
        self.model_path = model_path
        self.scaler_path = scaler_path

        # Load model and scaler
        self.load_model()

        # Initialize feature extractor
        self.extractor = FeatureExtractor(
            dataset_path="dummy",
            classes=self.classes,
            n_jobs=1,
        )

        # Prediction smoothing (moving average)
        self.prediction_buffer = deque(maxlen=5)  # Last 5 predictions
        self.confidence_buffer = deque(maxlen=5)
        self.show_overlay = True
        self.overlay_alpha = overlay_alpha

        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()

        # Color mapping for each class
        self.class_colors = {
            "cardboard": (139, 69, 19),  # Brown
            "glass": (0, 255, 255),  # Cyan
            "metal": (192, 192, 192),  # Silver
            "paper": (255, 255, 255),  # White
            "plastic": (0, 165, 255),  # Orange
            "trash": (128, 128, 128),  # Gray
        }

    def load_model(self):
        """Load trained model and scaler"""
        print("Loading model...")

        # Load model
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.classes = model_data["classes"]
        self.conf_threshold = model_data.get("confidence_threshold", 0.4)

        # Try to load decision margin threshold for SVM
        if hasattr(self.model, "decision_function"):
            self.margin_threshold = model_data.get("decision_margin_threshold", 0.5)
            self.use_svm = True
        else:
            self.distance_ratio_threshold = model_data.get(
                "distance_ratio_threshold", 2.5
            )
            self.distance_stats = model_data.get("distance_stats", None)
            self.use_svm = False

        # Load scaler
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        model_type = "SVM" if self.use_svm else "k-NN"
        print(f"✓ {model_type} model loaded successfully!")
        print(f"  Classes: {self.classes}")
        print(f"  Confidence threshold: {self.conf_threshold}")

    def extract_features(self, image):
        """Extract features from image"""
        try:
            features = self.extractor.extract_features(image)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def predict(self, features):
        """Make prediction with confidence scoring"""
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Get predictions
        probas = self.model.predict_proba(features_scaled)
        max_proba = probas.max()
        prediction = self.model.predict(features_scaled)[0]

        # Get all class probabilities
        prob_dict = {}
        for i, cls in enumerate(self.classes):
            if i < probas.shape[1]:
                prob_dict[cls] = float(probas[0][i])

        # Check confidence for SVM
        if self.use_svm:
            decision_values = self.model.decision_function(features_scaled)
            if decision_values.ndim == 1:
                margin = abs(decision_values[0])
            else:
                sorted_scores = np.sort(decision_values[0])
                margin = sorted_scores[-1] - sorted_scores[-2]

            rejected = max_proba < self.conf_threshold or margin < self.margin_threshold
        else:
            # k-NN distance-based rejection
            distances, _ = self.model.kneighbors(features_scaled)
            avg_distance = distances.mean()

            if self.distance_stats:
                reference_distance = self.distance_stats["median"]
                distance_ratio = avg_distance / reference_distance
            else:
                distance_ratio = avg_distance

            rejected = (
                max_proba < self.conf_threshold
                or distance_ratio > self.distance_ratio_threshold
            )

        return {
            "class_id": int(prediction),
            "class_label": self.classes[prediction],
            "confidence": float(max_proba),
            "rejected": rejected,
            "probabilities": prob_dict,
        }

    def smooth_prediction(self, result):
        """Smooth predictions using moving average"""
        self.prediction_buffer.append(result["class_label"])
        self.confidence_buffer.append(result["confidence"])

        # Most common prediction in buffer
        from collections import Counter

        most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
        avg_confidence = np.mean(self.confidence_buffer)

        return most_common, avg_confidence

    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)

    def draw_ui(self, frame, result, smoothed_class, smoothed_conf, fps):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()

        # Top bar - Title
        cv2.rectangle(overlay, (0, 0), (w, 60), (50, 50, 50), -1)
        cv2.putText(
            overlay,
            "REAL-TIME WASTE CLASSIFIER",
            (20, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # FPS counter
        cv2.putText(
            overlay,
            f"FPS: {fps:.1f}",
            (w - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Bottom panel - Results (compact to keep more camera area visible)
        panel_height = 170
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (50, 50, 50), -1)

        # Main prediction (smoothed)
        class_color = self.class_colors.get(smoothed_class, (255, 255, 255))

        if result["rejected"]:
            display_text = "UNKNOWN / LOW CONFIDENCE"
            class_color = (0, 0, 255)  # Red
        else:
            display_text = smoothed_class.upper()

        cv2.putText(
            overlay,
            "Classification:",
            (20, h - 210),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            display_text,
            (20, h - 165),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            class_color,
            3,
        )

        # Confidence
        cv2.putText(
            overlay,
            f"Confidence: {smoothed_conf*100:.1f}%",
            (20, h - 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Top 3 probabilities
        sorted_probs = sorted(
            result["probabilities"].items(), key=lambda x: x[1], reverse=True
        )[:3]
        y_offset = h - 100
        for cls, prob in sorted_probs:
            color = self.class_colors.get(cls, (255, 255, 255))
            bar_width = int(300 * prob)

            # Draw probability bar
            cv2.rectangle(
                overlay, (20, y_offset), (20 + bar_width, y_offset + 20), color, -1
            )
            cv2.rectangle(
                overlay, (20, y_offset), (320, y_offset + 20), (100, 100, 100), 2
            )

            # Label
            cv2.putText(
                overlay,
                f"{cls}: {prob*100:.1f}%",
                (330, y_offset + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 30

        # Instructions
        cv2.putText(
            overlay,
            "Press 'q' to quit | 's' save | SPACE pause | 'h' toggle UI",
            (20, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Blend overlay with original frame
        cv2.addWeighted(
            overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame
        )

        return frame

    def run(self, camera_index=0, skip_frames=2, fullscreen=False):
        """
        Run real-time classification

        Args:
            camera_index: Camera device index (0 for default webcam)
            skip_frames: Process every Nth frame (improves performance)
            fullscreen: Show window fullscreen (True/False)
        """
        # Open camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("\n" + "=" * 70)
        print("REAL-TIME WASTE CLASSIFICATION STARTED")
        print("=" * 70)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  SPACE - Pause/Resume")
        print("=" * 70 + "\n")

        frame_count = 0
        paused = False
        last_result = None
        smoothed_class = "Initializing..."
        smoothed_conf = 0.0

        window_name = "Real-Time Waste Classifier"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            # Allow manual resize; start maximized-ish by enlarging window
            cv2.resizeWindow(window_name, 1280, 800)

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame")
                        break

                    # Mirror the frame for more intuitive interaction
                    frame = cv2.flip(frame, 1)

                    # Process every Nth frame to improve performance
                    if frame_count % skip_frames == 0:
                        # Extract features and predict
                        features = self.extract_features(frame)

                        if features is not None:
                            result = self.predict(features)
                            smoothed_class, smoothed_conf = self.smooth_prediction(
                                result
                            )
                            last_result = result

                    # Calculate FPS
                    fps = self.calculate_fps()

                    # Draw UI
                    if last_result is not None and self.show_overlay:
                        frame = self.draw_ui(
                            frame, last_result, smoothed_class, smoothed_conf, fps
                        )

                    frame_count += 1

                # Display frame
                cv2.imshow(window_name, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nQuitting...")
                    break
                elif key == ord("s"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}_{smoothed_class}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord(" "):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord("h"):
                    self.show_overlay = not self.show_overlay
                    print(
                        "Overlay hidden" if not self.show_overlay else "Overlay shown"
                    )

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("\nCamera released. Goodbye!")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time waste classification from webcam"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_model/best_model.pkl",
        help="Path to trained model (default: best_model/best_model.pkl)",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="features/scaler.pkl",
        help="Path to scaler (default: features/scaler.pkl)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Process every Nth frame (default: 2, higher = faster but less responsive)",
    )
    parser.add_argument(
        "--fullscreen", action="store_true", help="Open display in fullscreen"
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.65,
        help="Overlay transparency (0-1, lower shows more camera)",
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("\nAvailable models:")
        for pkl in Path(".").rglob("*_model.pkl"):
            print(f"  - {pkl}")
        return

    # Initialize and run classifier
    classifier = RealtimeWasteClassifier(
        model_path=args.model,
        scaler_path=args.scaler,
        overlay_alpha=args.overlay_alpha,
    )
    classifier.run(
        camera_index=args.camera,
        skip_frames=args.skip_frames,
        fullscreen=args.fullscreen,
    )


if __name__ == "__main__":
    main()
