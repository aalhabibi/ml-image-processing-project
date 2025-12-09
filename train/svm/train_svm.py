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
    """Support Vector Machine Classifier Trainer"""

    def __init__(self, features_path="processed_features.pkl", save_dir="./train/svm"):
        self.features_path = features_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.best_params = None
        self.classes = None
        self.results = {}

        print("\n" + "=" * 70)
        print("SUPPORT VECTOR MACHINE (SVM) CLASSIFIER TRAINING")
        print("=" * 70)

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
            "C": [1, 10, 50, 100],
            "gamma": ["scale", "auto", 0.001],
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

        svm = SVC(random_state=42, max_iter=10000, class_weight="balanced")
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

        return grid_search.best_estimator_

    def train(self, X_train, y_train, use_best_params=True):
        """Train SVM"""
        print("\n" + "-" * 70)
        print("[3/4] TRAINING SVM CLASSIFIER")
        print("-" * 70)

        if use_best_params and self.best_params:
            print("\nUsing optimized parameters")
            self.model = SVC(**self.best_params, random_state=42, max_iter=10000)
        else:
            print("\nUsing default: C=1.0, kernel=rbf, gamma=scale")
            self.model = SVC(random_state=42, max_iter=10000)

        start_time = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start_time

        print(f"Training completed in {elapsed:.4f}s")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n" + "-" * 70)
        print("[4/4] MODEL EVALUATION")
        print("-" * 70)

        # Predict
        print("\nPredicting on test set...")
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        elapsed = time.time() - start_time

        avg_time = elapsed / len(X_test) * 1000
        print(f"Prediction time: {elapsed:.3f}s ({avg_time:.2f}ms per sample)")

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted"
        )

        print(f"\n{'='*70}")
        print("PERFORMANCE METRICS")
        print(f"{'='*70}")
        print(f"Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1-Score:   {f1:.4f}")

        # Per-class report
        print(f"\n{'='*70}")
        print("PER-CLASS PERFORMANCE")
        print(f"{'='*70}")
        print(
            classification_report(y_test, y_pred, target_names=self.classes, digits=4)
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, self.classes)

        # Store results
        self.results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "best_params": self.best_params,
            "confusion_matrix": cm.tolist(),
            "classes": self.classes,
            "avg_inference_time_ms": float(avg_time),
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
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))

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
            "SVM Confusion Matrix (Normalized)", fontsize=14, fontweight="bold", pad=20
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
    accuracy, _ = trainer.evaluate(X_test, y_test)

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
