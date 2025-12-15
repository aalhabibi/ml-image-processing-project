from train.knn.train_knn import KNNTrainer
from train.svm.train_svm import SVMTrainer
from .data_loader import DatasetInfo
from .augmentation import Augmentor
from .feature_extraction import FeatureExtractor


class WastePipeline:
    def __init__(
        self,
        dataset_path="data/dataset",
        train_split_path="data/train_split",
        test_split_path="data/test_split",
        augmented_train_path="data/augmented_train",
    ):
        self.dataset_path = dataset_path
        self.train_split_path = train_split_path
        self.test_split_path = test_split_path
        self.augmented_train_path = augmented_train_path

    def run(self):
        """
        CORRECTED PIPELINE - Prevents Data Leakage!

        Old (WRONG) approach:
          1. Augment entire dataset
          2. Split augmented data → DATA LEAKAGE!

        New (CORRECT) approach:
          1. Split original dataset first
          2. Augment ONLY training split
          3. Keep test split pristine
          4. Extract features separately
          5. Train without additional splitting
        """
        print("\n" + "=" * 70)
        print("WASTE CLASSIFICATION PIPELINE (CORRECTED)")
        print("=" * 70)
        print("\n✓ This pipeline prevents data leakage by:")
        print("  1. Splitting BEFORE augmentation")
        print("  2. Augmenting ONLY the training set")
        print("  3. Keeping test set pristine (no augmented copies)")
        print("=" * 70)

        # 1. Load dataset info
        print("\n" + "=" * 70)
        print("STEP 1: Load Original Dataset")
        print("=" * 70)
        info = DatasetInfo(self.dataset_path)
        classes, counts = info.load_dataset_info()

        # 2. SPLIT FIRST (CRITICAL!)
        print("\n" + "=" * 70)
        print("STEP 2: Split Into Train/Test (BEFORE Augmentation)")
        print("=" * 70)
        split_stats = info.split_dataset(
            train_path=self.train_split_path,
            test_path=self.test_split_path,
            test_size=0.2,
            random_state=42,
        )

        # 3. Augment ONLY training data
        print("\n" + "=" * 70)
        print("STEP 3: Augment TRAINING Set Only")
        print("=" * 70)
        augmentor = Augmentor(
            self.train_split_path,  # Only augment train split!
            self.augmented_train_path,
            classes,
        )
        aug_stats = augmentor.perform_augmentation(target_count=1000)

        # 4. Feature extraction from split datasets
        print("\n" + "=" * 70)
        print("STEP 4: Extract Features (Separately from Train/Test)")
        print("=" * 70)
        extractor = FeatureExtractor(
            self.augmented_train_path,  # Will be overridden in extract_features_split
            classes,
            n_jobs=-1,
        )
        X_train, X_test, y_train, y_test = extractor.extract_features_split(
            train_path=self.augmented_train_path,
            test_path=self.test_split_path,  # Pristine test data!
        )

        # 5. Train k-NN Classifier (NO SPLITTING - already split!)
        print("\n" + "=" * 70)
        print("STEP 5: Train k-NN Classifier")
        print("=" * 70)
        knn_trainer = KNNTrainer(features_path="features/train_features.pkl")
        knn_trainer.classes = classes

        # Load pre-split data
        X_train_knn, y_train_knn = X_train, y_train
        X_test_knn, y_test_knn = X_test, y_test

        best_model = knn_trainer.hyperparameter_tuning(X_train_knn, y_train_knn)
        knn_trainer.model = best_model
        knn_accuracy, _ = knn_trainer.evaluate(X_test_knn, y_test_knn)
        print(f"\n  k-NN Test Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
        knn_trainer.save_model("knn_model.pkl")

        # 6. Train SVM Classifier (NO SPLITTING - already split!)
        print("\n" + "=" * 70)
        print("STEP 6: Train SVM Classifier")
        print("=" * 70)
        svm_trainer = SVMTrainer(features_path="features/train_features.pkl")
        svm_trainer.classes = classes

        # Load pre-split data
        X_train_svm, y_train_svm = X_train, y_train
        X_test_svm, y_test_svm = X_test, y_test

        best_svm = svm_trainer.hyperparameter_tuning(X_train_svm, y_train_svm)
        svm_trainer.model = best_svm
        svm_accuracy, _ = svm_trainer.evaluate(X_test_svm, y_test_svm)
        print(f"\n  SVM Test Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        svm_trainer.save_model("svm_model.pkl")

        # Final comparison
        print("\n" + "=" * 70)
        print("FINAL RESULTS (NO DATA LEAKAGE)")
        print("=" * 70)
        print(f"k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
        print(f"SVM Accuracy:  {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        best = "SVM" if svm_accuracy > knn_accuracy else "k-NN"
        best_acc = max(svm_accuracy, knn_accuracy)
        print(f"\nBest Model: {best} with {best_acc:.4f} ({best_acc*100:.2f}%)")

        print("\n" + "=" * 70)
        print("⚠️  IMPORTANT NOTE")
        print("=" * 70)
        print("These accuracies are REAL - no data leakage!")
        print("  ✓ Test set contains NO augmented versions of training images")
        print("  ✓ Model has never seen test images in any form during training")
        print("  ✓ This is the true generalization performance")

        if knn_accuracy < 0.7 or svm_accuracy < 0.7:
            print("\n⚠️  Lower than expected? This is GOOD!")
            print("   Your previous high accuracy was likely inflated by data leakage.")
            print("   Now you're seeing the model's TRUE performance.")

        print("\nPipeline complete!")
        return {
            "knn_accuracy": knn_accuracy,
            "svm_accuracy": svm_accuracy,
            "best_model": best,
            "split_stats": split_stats,
            "aug_stats": aug_stats,
        }
