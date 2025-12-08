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
    """K-Nearest Neighbors Classifier Trainer"""

    def __init__(self, features_path="processed_features.pkl"):
        self.features_path = features_path
        self.model = None
        self.best_params = None
        self.classes = None
        self.results = {}

        print("\n" + "=" * 70)
        print("K-NEAREST NEIGHBORS (k-NN) CLASSIFIER TRAINING")
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
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
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

        return grid_search.best_estimator_

    def train(self, X_train, y_train, use_best_params=True):
        """Train k-NN (stores training data)"""
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

    def save_model(self, filename="knn_model.pkl"):
        """Save trained model"""
        print("\nSaving model...")

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
        json_file = filename.replace(".pkl", "_results.json")
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
        plt.savefig("knn_k_selection.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved: knn_k_selection.png")
        plt.close()

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
            "k-NN Confusion Matrix (Normalized)", fontsize=14, fontweight="bold", pad=20
        )
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        plt.savefig("knn_confusion_matrix.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved: knn_confusion_matrix.png")
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
    accuracy, _ = trainer.evaluate(X_test, y_test)

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
