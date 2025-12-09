#!/usr/bin/env python3
"""
predict_knn.py - KNN Model Inference with Unknown Class Detection
Demonstrates how to use the trained KNN model with rejection mechanism
"""

import pickle
import numpy as np
from pathlib import Path


class KNNPredictor:
    """KNN Classifier with Unknown Class Detection"""

    def __init__(self, model_path="./train/knn/knn_model.pkl"):
        """Load trained KNN model with rejection parameters"""
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print("Loading KNN model with rejection mechanism...")
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.classes = data["classes"]
        self.confidence_threshold = data.get("confidence_threshold", 0.6)
        self.distance_threshold = data.get("distance_threshold", None)
        self.distance_stats = data.get("distance_stats", None)

        # Classes include 'unknown' now
        print(f"Model loaded successfully!")
        print(f"Classes: {self.classes}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Distance threshold: {self.distance_threshold}")

    def predict(self, X, return_details=False):
        """
        Predict with unknown class rejection

        Args:
            X: Feature vectors (n_samples, n_features) or single sample
            return_details: If True, return confidence and distance info

        Returns:
            predictions: Class labels (0-5: known classes, 6: unknown)
            If return_details=True, also returns (confidences, distances, labels)
        """
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get probability predictions
        probas = self.model.predict_proba(X)
        max_probas = probas.max(axis=1)
        predictions = self.model.predict(X)

        # Get distances to k nearest neighbors
        distances, _ = self.model.kneighbors(X)
        avg_distances = distances.mean(axis=1)

        # Apply rejection criteria
        low_confidence = max_probas < self.confidence_threshold
        too_far = avg_distances > self.distance_threshold
        reject_mask = low_confidence | too_far

        # Mark as unknown (class 6)
        predictions[reject_mask] = 6

        # Get class labels
        labels = [
            self.classes[p] if p < len(self.classes) else "unknown" for p in predictions
        ]

        if return_details:
            return predictions, max_probas, avg_distances, labels

        return predictions, labels

    def predict_single(self, feature_vector):
        """
        Predict single sample with detailed output

        Args:
            feature_vector: Single feature vector (1D array)

        Returns:
            dict with prediction details
        """
        pred, conf, dist, label = self.predict(feature_vector, return_details=True)

        pred_id = pred[0]
        label_str = label[0]
        confidence = conf[0]
        distance = dist[0]

        # Determine rejection reason
        rejected = pred_id == 6
        rejection_reason = []
        if rejected:
            if confidence < self.confidence_threshold:
                rejection_reason.append(
                    f"Low confidence ({confidence:.3f} < {self.confidence_threshold})"
                )
            if distance > self.distance_threshold:
                rejection_reason.append(
                    f"Too far from training data ({distance:.3f} > {self.distance_threshold:.3f})"
                )

        return {
            "class_id": int(pred_id),
            "class_label": label_str,
            "confidence": float(confidence),
            "avg_distance": float(distance),
            "rejected": rejected,
            "rejection_reason": (
                ", ".join(rejection_reason) if rejection_reason else None
            ),
        }


def demo_prediction():
    """Demonstration of model usage"""
    print("\n" + "=" * 70)
    print("KNN MODEL INFERENCE DEMO - With Unknown Class Detection")
    print("=" * 70)

    # Load model
    predictor = KNNPredictor(model_path="./train/knn/knn_model.pkl")

    # Load test features for demonstration
    print("\nLoading test data for demonstration...")
    with open("features/processed_features.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]

    # Test on a few samples
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    # Test 5 random samples
    indices = np.random.choice(len(X), 5, replace=False)

    for i, idx in enumerate(indices, 1):
        sample = X[idx]
        true_label = predictor.classes[y[idx]]

        result = predictor.predict_single(sample)

        print(f"\nSample {i}:")
        print(f"  True class:      {true_label} (ID: {y[idx]})")
        print(f"  Predicted class: {result['class_label']} (ID: {result['class_id']})")
        print(f"  Confidence:      {result['confidence']:.4f}")
        print(f"  Avg distance:    {result['avg_distance']:.4f}")
        print(f"  Rejected:        {result['rejected']}")
        if result["rejected"]:
            print(f"  Reason:          {result['rejection_reason']}")

        # Status indicator
        if result["rejected"]:
            status = "⚠️  UNKNOWN/REJECTED"
        elif result["class_id"] == y[idx]:
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        print(f"  Status:          {status}")

    print("\n" + "=" * 70)
    print("To use this model in production:")
    print("  1. Load model with KNNPredictor(model_path)")
    print("  2. Extract features from new image")
    print("  3. Call predictor.predict_single(features)")
    print("  4. Check result['rejected'] before using prediction")
    print("=" * 70)


if __name__ == "__main__":
    try:
        demo_prediction()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train the model first:")
        print("   python train/knn/train_knn.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
