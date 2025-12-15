#!/usr/bin/env python3
"""
train_svm.py - Support Vector Machine Training Script
Standalone script for training SVM classifier on waste classification dataset
"""

import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
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


class SVMTrainer:
    """Support Vector Machine Classifier Trainer with Unknown Class Rejection"""

    def __init__(
        self,
        features_path="processed_features.pkl",
        save_dir="./train/svm",
        confidence_threshold=0.4,
        decision_margin_threshold=0.5,
    ):
        """
        Args:
            features_path: Path to preprocessed features
            save_dir: Directory to save model and results
            confidence_threshold: Minimum probability for prediction (0-1).
                                If max probability < threshold, classify as 'unknown'
                                Default: 0.4 (relaxed for better recall)
            decision_margin_threshold: Minimum distance from decision boundary.
                                     If margin < threshold, classify as 'unknown'
                                     Default: 0.5
        """
        self.features_path = features_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.best_params = None
        self.classes = None
        self.results = {}
        self.confidence_threshold = confidence_threshold
        self.decision_margin_threshold = decision_margin_threshold
        self.margin_stats = None  # Will store training margin statistics

        print("\n" + "=" * 70)
        print("SUPPORT VECTOR MACHINE (SVM) CLASSIFIER TRAINING")
        print("With Unknown Class Rejection Mechanism")
        print("=" * 70)
        print(f"\nRejection Parameters:")
        print(f"  Confidence Threshold: {confidence_threshold} (min probability)")
        print(
            f"  Decision Margin Threshold: {decision_margin_threshold} (min distance from boundary)"
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

    def hyperparameter_tuning(self, X_train, y_train):
        """Grid search for best hyperparameters"""
        print("\n" + "-" * 70)
        print("[2/4] HYPERPARAMETER TUNING (Grid Search)")
        print("-" * 70)

        param_grid = {
            "C": [10, 100],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf"],
        }

        print("\nParameter grid:")
        for param, values in param_grid.items():
            print(f"  {param:15s}: {values}")

        total = (
            len(param_grid["C"]) * len(param_grid["gamma"]) * len(param_grid["kernel"])
        )
        print(f"\nTotal combinations: {total}")
        print("This may take several minutes...")
        print("Progress will be shown below:\n")

        svm = SVC(
            probability=True, random_state=42, max_iter=10000, class_weight="balanced"
        )
        grid_search = GridSearchCV(
            estimator=svm,
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
                f"  {i}. {score:.4f} | C={params.get('C')}, kernel={params.get('kernel')}, gamma={params.get('gamma')}"
            )

        # Store the best model and compute margin statistics
        self.model = grid_search.best_estimator_
        self._compute_margin_statistics(X_train, y_train)

        return grid_search.best_estimator_

    def train(self, X_train, y_train, use_best_params=True):
        """Train SVM"""
        print("\n" + "-" * 70)
        print("[3/4] TRAINING SVM CLASSIFIER")
        print("-" * 70)

        if use_best_params and self.best_params:
            print("\nUsing optimized parameters")
            self.model = SVC(
                **self.best_params, probability=True, random_state=42, max_iter=10000
            )
        else:
            print("\nUsing default: C=1.0, kernel=rbf, gamma=scale")
            self.model = SVC(probability=True, random_state=42, max_iter=10000)

        start_time = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start_time

        print(f"Training completed in {elapsed:.4f}s")

        # Compute margin statistics for rejection mechanism
        self._compute_margin_statistics(X_train, y_train)

    def _compute_margin_statistics(self, X_train, y_train):
        """Compute decision function margin statistics for rejection mechanism"""
        print("\nComputing decision margin statistics for rejection mechanism...")

        # Sample a subset for efficiency (if dataset is large)
        n_samples = min(1000, len(X_train))
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train[indices]

        # Get decision function values (distance from decision boundary)
        decision_values = self.model.decision_function(X_sample)

        # For each sample, get the margin (difference between top 2 scores)
        if decision_values.ndim == 1:  # Binary classification
            margins = np.abs(decision_values)
        else:  # Multi-class
            sorted_scores = np.sort(decision_values, axis=1)
            margins = (
                sorted_scores[:, -1] - sorted_scores[:, -2]
            )  # Top score - second score

        # Store statistics
        self.margin_stats = {
            "mean": float(np.mean(margins)),
            "std": float(np.std(margins)),
            "median": float(np.median(margins)),
            "q25": float(np.percentile(margins, 25)),
            "q10": float(np.percentile(margins, 10)),
            "q05": float(np.percentile(margins, 5)),
        }

        print(f"  Median margin (training): {self.margin_stats['median']:.4f}")
        print(f"  25th percentile: {self.margin_stats['q25']:.4f}")
        print(f"  10th percentile: {self.margin_stats['q10']:.4f}")
        print(f"  Decision margin threshold: {self.decision_margin_threshold}")

    def predict_with_rejection(self, X, return_confidence=False):
        """
        Predict with rejection mechanism using decision margins

        Returns predictions where:
        - 0-5: Known classes (glass, paper, cardboard, plastic, metal, trash)
        - 6: Unknown (rejected samples)

        Args:
            X: Feature vectors to predict
            return_confidence: If True, also return confidence scores

        Returns:
            predictions: Array of class predictions (0-6)
            confidences: (Optional) Array of confidence scores and margins
        """
        # Ensure margin stats are computed
        if self.margin_stats is None:
            raise RuntimeError(
                "Margin statistics not computed. Please call train() or ensure "
                "the model has been properly initialized with margin statistics."
            )

        # Get probability predictions
        probas = self.model.predict_proba(X)
        max_probas = probas.max(axis=1)
        predictions = self.model.predict(X)

        # Get decision function values (distance from decision boundary)
        decision_values = self.model.decision_function(X)

        # Calculate decision margins (difference between top 2 scores)
        if decision_values.ndim == 1:  # Binary classification
            margins = np.abs(decision_values)
        else:  # Multi-class
            sorted_scores = np.sort(decision_values, axis=1)
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]

        # Apply rejection criteria
        # Criterion 1: Low confidence (probability)
        low_confidence = max_probas < self.confidence_threshold

        # Criterion 2: Close to decision boundary (small margin)
        close_to_boundary = margins < self.decision_margin_threshold

        # Mark as unknown (class 6) if either criterion fails
        reject_mask = low_confidence | close_to_boundary
        predictions[reject_mask] = 6  # Unknown class

        if return_confidence:
            confidences = max_probas.copy()
            confidences[reject_mask] = 0.0  # Zero confidence for rejected samples
            return predictions, confidences, margins

        return predictions

    def evaluate(self, X_test, y_test):
        """Evaluate model performance with rejection mechanism"""
        print("\n" + "-" * 70)
        print("[4/4] MODEL EVALUATION (with Unknown Class Rejection)")
        print("-" * 70)

        # Predict with rejection
        print("\nPredicting on test set with rejection mechanism...")
        start_time = time.time()
        y_pred, confidences, margins = self.predict_with_rejection(
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
        boundary_rejected = np.sum(
            (margins < self.decision_margin_threshold) & (y_pred == 6)
        )

        print(f"\nRejection Statistics:")
        print(
            f"  Samples rejected as 'unknown': {n_rejected}/{len(y_pred)} ({rejection_rate:.2f}%)"
        )
        if n_rejected > 0:
            print(f"    - Low confidence: {low_conf_rejected} samples")
            print(f"    - Close to boundary: {boundary_rejected} samples")
        print(
            f"  Average confidence (accepted): {confidences[y_pred != 6].mean():.4f}"
            if (y_pred != 6).sum() > 0
            else "  Average confidence (accepted): N/A"
        )
        if n_rejected > 0:
            print(f"  Average margin (rejected): {margins[y_pred == 6].mean():.4f}")

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
            "decision_margin_threshold": float(self.decision_margin_threshold),
            "margin_stats": self.margin_stats,
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

    def save_model(self, filename="./train/svm/svm_model.pkl"):
        """Save trained model"""
        print("\nSaving model...")
        filename = self.save_dir / "svm_model.pkl"

        model_data = {
            "model": self.model,
            "classes": self.classes,
            "best_params": self.best_params,
            "results": self.results,
            "confidence_threshold": self.confidence_threshold,
            "decision_margin_threshold": self.decision_margin_threshold,
            "margin_stats": self.margin_stats,
        }

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved: {filename}")

        # Save results JSON
        json_file = json_file = self.save_dir / "svm_model_results.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved: {json_file}")

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
            "SVM Confusion Matrix with Unknown Class (Normalized)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            self.save_dir / "svm_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved: svm_confusion_matrix.png")
        plt.close()

    def plot_rejection_analysis(self, X_test, y_test, y_pred, confidences, margins):
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

        # 2. Decision margin distribution
        ax = axes[0, 1]
        accepted_margin = margins[y_pred != 6]
        rejected_margin = margins[y_pred == 6]
        ax.hist(
            [accepted_margin, rejected_margin],
            bins=30,
            label=["Accepted", "Rejected"],
            alpha=0.7,
            color=["green", "red"],
        )
        ax.axvline(
            self.decision_margin_threshold,
            color="black",
            linestyle="--",
            label=f"Threshold={self.decision_margin_threshold:.2f}",
        )
        ax.set_xlabel("Decision Margin")
        ax.set_ylabel("Frequency")
        ax.set_title("Decision Margin Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Confidence vs Margin scatter
        ax = axes[1, 0]
        scatter_accepted = ax.scatter(
            margins[y_pred != 6],
            confidences[y_pred != 6],
            c="green",
            alpha=0.5,
            s=20,
            label="Accepted",
        )
        if len(rejected) > 0:
            scatter_rejected = ax.scatter(
                margins[y_pred == 6],
                confidences[y_pred == 6],
                c="red",
                alpha=0.5,
                s=20,
                label="Rejected",
            )
        ax.axhline(self.confidence_threshold, color="blue", linestyle="--", alpha=0.5)
        ax.axvline(
            self.decision_margin_threshold, color="blue", linestyle="--", alpha=0.5
        )
        ax.set_xlabel("Decision Margin")
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
            self.save_dir / "svm_rejection_analysis.png", dpi=300, bbox_inches="tight"
        )
        print(f"Plot saved: svm_rejection_analysis.png")
        plt.close()


def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print(" " * 20 + "SVM TRAINING PIPELINE")
    print("=" * 70)

    # Initialize
    trainer = SVMTrainer(features_path="processed_features.pkl")

    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()

    # Hyperparameter tuning (recommended)
    best_model = trainer.hyperparameter_tuning(X_train, y_train)
    trainer.model = best_model

    # Evaluate
    accuracy, y_pred = trainer.evaluate(X_test, y_test)

    # Get confidence and margin info for additional analysis
    _, confidences, margins = trainer.predict_with_rejection(
        X_test, return_confidence=True
    )

    # Plot rejection analysis
    trainer.plot_rejection_analysis(X_test, y_test, y_pred, confidences, margins)

    # Save
    trainer.save_model("svm_model.pkl")

    # Summary
    print("\n" + "=" * 70)
    print("SVM TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nGenerated files:")
    print("  - svm_model.pkl")
    print("  - svm_model_results.json")
    print("  - svm_confusion_matrix.png")


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
