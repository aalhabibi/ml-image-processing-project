#!/usr/bin/env python3
"""
train_knn.py - K-Nearest Neighbors Training Script
Standalone script for training k-NN classifier on waste classification dataset
"""

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from pathlib import Path
from tqdm import tqdm


class KNNTrainer:
    """K-Nearest Neighbors Classifier Trainer with Unknown Class Rejection"""

    def __init__(
        self,
        features_path="processed_features.pkl",
        save_dir="./train/knn",
        confidence_threshold=0.4,
        distance_ratio_threshold=2.5,
    ):
        """
        Args:
            features_path: Path to preprocessed features
            save_dir: Directory to save model and results
            confidence_threshold: Minimum probability for prediction (0-1).
                                If max probability < threshold, classify as 'unknown'
                                Default: 0.4 (relaxed for better recall)
            distance_ratio_threshold: Ratio of sample distance to typical training distance.
                                    If ratio > threshold, classify as 'unknown'
                                    Default: 2.5 (sample is 2.5x further than typical)
        """
        self.features_path = features_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.best_params = None
        self.classes = None
        self.results = {}
        self.confidence_threshold = confidence_threshold
        self.distance_ratio_threshold = distance_ratio_threshold
        self.distance_stats = None  # Will store training distance statistics

        print("\n" + "=" * 70)
        print("K-NEAREST NEIGHBORS (k-NN) CLASSIFIER TRAINING")
        print("With Unknown Class Rejection Mechanism (Improved)")
        print("=" * 70)
        print(f"\nRejection Parameters:")
        print(f"  Confidence Threshold: {confidence_threshold} (min probability)")
        print(
            f"  Distance Ratio Threshold: {distance_ratio_threshold}x (relative to training)"
        )

    def load_data(self):
        """Load preprocessed features"""
        print("\n[1/4] Loading preprocessed features...")

        if not Path(self.features_path).exists():
            raise FileNotFoundError(
                f"Features file not found: {self.features_path}\n"
                "Please run feature extraction first!"
            )

        with open(self.features_path, "rb") as f:
            data = pickle.load(f)

        X = data["X"]
        y = data["y"]
        self.classes = data["classes"]
        feature_dim = data["feature_dim"]

        print(f"Loaded {len(X)} samples")
        print(f"Feature dimension: {feature_dim}")
        print(f"Classes: {self.classes}")

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nDataset split:")
        print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

        # Class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining set distribution:")
        for cls_idx, count in zip(unique, counts):
            print(f"  {self.classes[cls_idx]:12s}: {count:4d} samples")

        return X_train, X_test, y_train, y_test

    def find_best_k(self, X_train, y_train, k_range=None):
        """Find optimal k using cross-validation"""
        print("\n" + "-" * 70)
        print("FINDING OPTIMAL K VALUE")
        print("-" * 70)

        if k_range is None:
            max_k = int(np.sqrt(len(X_train)))
            k_range = range(1, min(max_k, 50), 2)

        print(f"Testing k values: {list(k_range)}")
        print("Using 5-fold cross-validation...\n")

        scores = []
        best_k = 1
        best_score = 0

        for k in tqdm(k_range, desc="Testing k values"):
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            cv_scores = cross_val_score(knn, X_train, y_train, cv=5, n_jobs=-1)
            mean_score = cv_scores.mean()
            scores.append(mean_score)

            if mean_score > best_score:
                best_score = mean_score
                best_k = k

        print(f"\nBest k: {best_k} (accuracy: {best_score:.4f})")

        # Plot
        self._plot_k_selection(list(k_range), scores, best_k)

        return best_k, scores

    def hyperparameter_tuning(self, X_train, y_train):
        """Grid search for best hyperparameters"""
        print("\n" + "-" * 70)
        print("[2/4] HYPERPARAMETER TUNING (Grid Search)")
        print("-" * 70)

        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["distance"],
            "metric": ["manhattan"],
        }

        print("\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param:15s}: {values}")

        total = np.prod([len(v) for v in param_grid.values()])
        print(f"\nTotal combinations: {total}")
        print("This may take several minutes...")
        print("Progress will be shown below:\n")

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            verbose=2,
            n_jobs=4,
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start_time

        print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param:15s}: {value}")
        print(f"\nBest CV accuracy: {grid_search.best_score_:.4f}")

        self.best_params = grid_search.best_params_

        # Top 5 configs
        results = grid_search.cv_results_
        top_idx = np.argsort(results["mean_test_score"])[-5:][::-1]

        print("\nTop 5 configurations:")
        for i, idx in enumerate(top_idx, 1):
            score = results["mean_test_score"][idx]
            params = results["params"][idx]
            print(
                f"  {i}. {score:.4f} | k={params['n_neighbors']}, "
                f"weights={params['weights']}, metric={params['metric']}"
            )

        # Store the best model and compute distance statistics
        self.model = grid_search.best_estimator_
        self._compute_distance_statistics(X_train)

        return grid_search.best_estimator_

    def train(self, X_train, y_train, use_best_params=True):
        """Train k-NN (stores training data) and compute distance statistics"""
        print("\n" + "-" * 70)
        print("[3/4] TRAINING k-NN CLASSIFIER")
        print("-" * 70)

        if use_best_params and self.best_params:
            print("\nUsing optimized parameters")
            self.model = KNeighborsClassifier(**self.best_params, n_jobs=-1)
        else:
            print("\nUsing default: k=5, uniform, euclidean")
            self.model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

        start_time = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start_time

        print(f"Training completed in {elapsed:.4f}s")

        # Compute distance statistics for rejection mechanism
        self._compute_distance_statistics(X_train)

    def _compute_distance_statistics(self, X_train):
        """Compute distance statistics using relative ratios instead of absolute values"""
        print("\nComputing distance statistics for rejection mechanism...")

        # Sample a subset for efficiency (if dataset is large)
        n_samples = min(1000, len(X_train))
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sample = X_train[indices]

        # Get distances to k nearest neighbors
        distances, _ = self.model.kneighbors(X_sample)
        avg_distances = distances.mean(axis=1)

        # Store statistics - using median as reference (more robust than mean)
        self.distance_stats = {
            "mean": float(np.mean(avg_distances)),
            "std": float(np.std(avg_distances)),
            "median": float(np.median(avg_distances)),
            "q75": float(np.percentile(avg_distances, 75)),
            "q90": float(np.percentile(avg_distances, 90)),
            "q95": float(np.percentile(avg_distances, 95)),
        }

        print(f"  Median distance (training): {self.distance_stats['median']:.4f}")
        print(f"  75th percentile: {self.distance_stats['q75']:.4f}")
        print(f"  90th percentile: {self.distance_stats['q90']:.4f}")
        print(f"  Distance ratio threshold: {self.distance_ratio_threshold}x median")

    def predict_with_rejection(self, X, return_confidence=False):
        """
        Predict with rejection mechanism using RELATIVE distance ratios

        Returns predictions where:
        - 0-5: Known classes (glass, paper, cardboard, plastic, metal, trash)
        - 6: Unknown (rejected samples)

        Args:
            X: Feature vectors to predict
            return_confidence: If True, also return confidence scores

        Returns:
            predictions: Array of class predictions (0-6)
            confidences: (Optional) Array of confidence scores
        """
        # Ensure distance stats are computed
        if self.distance_stats is None:
            raise RuntimeError(
                "Distance statistics not computed. Please call train() or ensure "
                "the model has been properly initialized with distance statistics."
            )

        # Get probability predictions
        probas = self.model.predict_proba(X)
        max_probas = probas.max(axis=1)
        predictions = self.model.predict(X)

        # Get distances to k nearest neighbors
        distances, _ = self.model.kneighbors(X)
        avg_distances = distances.mean(axis=1)

        # Use RELATIVE distance ratio instead of absolute threshold
        # Compare each sample's distance to the typical training distance (median)
        reference_distance = self.distance_stats["median"]
        distance_ratios = avg_distances / reference_distance

        # Apply rejection criteria
        # Criterion 1: Low confidence (probability)
        low_confidence = max_probas < self.confidence_threshold

        # Criterion 2: Too far from training data (relative distance)
        too_far = distance_ratios > self.distance_ratio_threshold

        # Mark as unknown (class 6) if either criterion fails
        reject_mask = low_confidence | too_far
        predictions[reject_mask] = 6  # Unknown class

        if return_confidence:
            confidences = max_probas.copy()
            confidences[reject_mask] = 0.0  # Zero confidence for rejected samples
            return predictions, confidences, avg_distances, distance_ratios

        return predictions

    def evaluate(self, X_test, y_test):
        """Evaluate model performance with rejection mechanism"""
        print("\n" + "-" * 70)
        print("[4/4] MODEL EVALUATION (with Unknown Class Rejection)")
        print("-" * 70)

        # Predict with rejection
        print("\nPredicting on test set with rejection mechanism...")
        start_time = time.time()
        y_pred, confidences, distances, distance_ratios = self.predict_with_rejection(
            X_test, return_confidence=True
        )
        elapsed = time.time() - start_time

        avg_time = elapsed / len(X_test) * 1000
        print(f"Prediction time: {elapsed:.3f}s ({avg_time:.2f}ms per sample)")

        # Rejection statistics
        n_rejected = np.sum(y_pred == 6)
        rejection_rate = n_rejected / len(y_pred) * 100

        # Analyze rejection reasons
        low_conf_rejected = np.sum((confidences == 0) & (y_pred == 6))
        far_rejected = np.sum(
            (distance_ratios > self.distance_ratio_threshold) & (y_pred == 6)
        )

        print(f"\nRejection Statistics:")
        print(
            f"  Samples rejected as 'unknown': {n_rejected}/{len(y_pred)} ({rejection_rate:.2f}%)"
        )
        if n_rejected > 0:
            print(f"    - Low confidence: {low_conf_rejected} samples")
            print(f"    - Too far (distance ratio): {far_rejected} samples")
        print(
            f"  Average confidence (accepted): {confidences[y_pred != 6].mean():.4f}"
            if (y_pred != 6).sum() > 0
            else "  Average confidence (accepted): N/A"
        )
        if n_rejected > 0:
            print(
                f"  Average distance ratio (rejected): {distance_ratios[y_pred == 6].mean():.4f}x median"
            )

        # Metrics (excluding unknown class for main metrics)
        # Only evaluate on samples that were not rejected
        accepted_mask = y_pred != 6
        y_test_accepted = y_test[accepted_mask]
        y_pred_accepted = y_pred[accepted_mask]

        accuracy = (
            accuracy_score(y_test_accepted, y_pred_accepted)
            if len(y_test_accepted) > 0
            else 0.0
        )
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test_accepted, y_pred_accepted, average="weighted", zero_division=0
            )
            if len(y_test_accepted) > 0
            else (0.0, 0.0, 0.0, None)
        )

        print(f"\n{'='*70}")
        print("PERFORMANCE METRICS (Accepted Samples Only)")
        print(f"{'='*70}")
        print(f"Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1-Score:   {f1:.4f}")

        # Per-class report (with unknown class)
        classes_with_unknown = self.classes + ["unknown"]
        print(f"\n{'='*70}")
        print("PER-CLASS PERFORMANCE (Including Unknown)")
        print(f"{'='*70}")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=classes_with_unknown,
                labels=list(range(len(classes_with_unknown))),
                digits=4,
                zero_division=0,
            )
        )

        # Confusion matrix
        cm = confusion_matrix(
            y_test, y_pred, labels=list(range(len(classes_with_unknown)))
        )
        self._plot_confusion_matrix(cm, classes_with_unknown)

        # Store results
        self.results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "best_params": self.best_params,
            "confusion_matrix": cm.tolist(),
            "classes": classes_with_unknown,
            "avg_inference_time_ms": float(avg_time),
            "rejection_rate": float(rejection_rate),
            "n_rejected": int(n_rejected),
            "confidence_threshold": float(self.confidence_threshold),
            "distance_ratio_threshold": float(self.distance_ratio_threshold),
            "distance_stats": self.distance_stats,
        }

        # Target check
        target = 0.85
        print(f"\n{'='*70}")
        if accuracy >= target:
            print(f"TARGET ACHIEVED: {accuracy:.4f} >= {target}")
        else:
            print(f"Target not met: {accuracy:.4f} < {target}")
            print(f"Gap: {(target - accuracy)*100:.2f}%")
        print(f"{'='*70}")

        return accuracy, y_pred

    def save_model(self, filename="knn_model.pkl"):
        """Save trained model with rejection parameters"""
        print("\nSaving model...")
        filename = self.save_dir / "knn_model.pkl"

        model_data = {
            "model": self.model,
            "classes": self.classes,
            "best_params": self.best_params,
            "results": self.results,
            "confidence_threshold": self.confidence_threshold,
            "distance_ratio_threshold": self.distance_ratio_threshold,
            "distance_stats": self.distance_stats,
        }

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved: {filename}")

        # Save results JSON
        json_file = self.save_dir / "knn_model_results.json"

        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved: {json_file}")

    def _plot_k_selection(self, k_values, scores, best_k):
        """Plot k vs accuracy"""
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, scores, "b-o", linewidth=2, markersize=8)
        plt.axvline(x=best_k, color="r", linestyle="--", label=f"Best k={best_k}")
        plt.xlabel("Number of Neighbors (k)", fontsize=12)
        plt.ylabel("Cross-Validation Accuracy", fontsize=12)
        plt.title("k-NN: K Value Selection", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(self.save_dir / "knn_k_selection.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved: knn_k_selection.png")
        plt.close()

    def _plot_confusion_matrix(self, cm, classes):
        """Plot confusion matrix with unknown class"""
        plt.figure(figsize=(11, 9))

        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"label": "Percentage"},
        )

        plt.title(
            "k-NN Confusion Matrix with Unknown Class (Normalized)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / "knn_confusion_matrix.png", dpi=300, bbox_inches="tight"
        )
        print(f"Plot saved: knn_confusion_matrix.png")
        plt.close()

    def plot_rejection_analysis(
        self, X_test, y_test, y_pred, confidences, distances, distance_ratios
    ):
        """Plot rejection mechanism analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Confidence distribution
        ax = axes[0, 0]
        accepted = confidences[y_pred != 6]
        rejected = confidences[y_pred == 6]
        ax.hist(
            [accepted, rejected],
            bins=30,
            label=["Accepted", "Rejected"],
            alpha=0.7,
            color=["green", "red"],
        )
        ax.axvline(
            self.confidence_threshold,
            color="black",
            linestyle="--",
            label=f"Threshold={self.confidence_threshold}",
        )
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Confidence Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Distance RATIO distribution (relative, not absolute)
        ax = axes[0, 1]
        accepted_ratio = distance_ratios[y_pred != 6]
        rejected_ratio = distance_ratios[y_pred == 6]
        ax.hist(
            [accepted_ratio, rejected_ratio],
            bins=30,
            label=["Accepted", "Rejected"],
            alpha=0.7,
            color=["green", "red"],
        )
        ax.axvline(
            self.distance_ratio_threshold,
            color="black",
            linestyle="--",
            label=f"Threshold={self.distance_ratio_threshold:.1f}x",
        )
        ax.set_xlabel("Distance Ratio (relative to median)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distance Ratio Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Confidence vs Distance Ratio scatter
        ax = axes[1, 0]
        scatter_accepted = ax.scatter(
            distance_ratios[y_pred != 6],
            confidences[y_pred != 6],
            c="green",
            alpha=0.5,
            s=20,
            label="Accepted",
        )
        if len(rejected) > 0:
            scatter_rejected = ax.scatter(
                distance_ratios[y_pred == 6],
                confidences[y_pred == 6],
                c="red",
                alpha=0.5,
                s=20,
                label="Rejected",
            )
        ax.axhline(self.confidence_threshold, color="blue", linestyle="--", alpha=0.5)
        ax.axvline(
            self.distance_ratio_threshold, color="blue", linestyle="--", alpha=0.5
        )
        ax.set_xlabel("Distance Ratio (relative to median)")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Rejection Decision Boundary")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Rejection rate by true class
        ax = axes[1, 1]
        classes_with_unknown = self.classes + ["unknown"]
        rejection_by_class = []
        for i in range(len(self.classes)):
            mask = y_test == i
            if mask.sum() > 0:
                rejection_rate = ((y_pred[mask] == 6).sum() / mask.sum()) * 100
                rejection_by_class.append(rejection_rate)
            else:
                rejection_by_class.append(0)

        bars = ax.bar(range(len(self.classes)), rejection_by_class, color="coral")
        ax.set_xlabel("True Class")
        ax.set_ylabel("Rejection Rate (%)")
        ax.set_title("Rejection Rate by True Class")
        ax.set_xticks(range(len(self.classes)))
        ax.set_xticklabels(self.classes, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            self.save_dir / "knn_rejection_analysis.png", dpi=300, bbox_inches="tight"
        )
        print(f"Plot saved: knn_rejection_analysis.png")
        plt.close()


def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print(" " * 20 + "k-NN TRAINING PIPELINE")
    print("=" * 70)

    # Initialize
    trainer = KNNTrainer(features_path="processed_features.pkl")

    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()

    # Hyperparameter tuning (recommended)
    best_model = trainer.hyperparameter_tuning(X_train, y_train)
    trainer.model = best_model

    # Alternative: Quick training without tuning
    # trainer.train(X_train, y_train, use_best_params=False)

    # Evaluate
    accuracy, y_pred = trainer.evaluate(X_test, y_test)

    # Get confidence and distance info for additional analysis
    _, confidences, distances, distance_ratios = trainer.predict_with_rejection(
        X_test, return_confidence=True
    )

    # Plot rejection analysis
    trainer.plot_rejection_analysis(
        X_test, y_test, y_pred, confidences, distances, distance_ratios
    )

    # Save
    trainer.save_model("knn_model.pkl")

    # Summary
    print("\n" + "=" * 70)
    print("k-NN TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nGenerated files:")
    print("  - knn_model.pkl")
    print("  - knn_model_results.json")
    print("  - knn_confusion_matrix.png")
    print("  - knn_k_selection.png")
    print("\nNext: Train SVM for comparison")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure to run feature extraction first:")
        print("   python feature_extraction.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
