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
        """Execute the split-first pipeline to avoid data leakage."""

        print("\n" + "=" * 70)
        print("STEP 1: Load Original Dataset")
        print("=" * 70)
        info = DatasetInfo(self.dataset_path)
        classes, counts = info.load_dataset_info()

        print("\n" + "=" * 70)
        print("STEP 2: Split Into Train/Test (BEFORE Augmentation)")
        print("=" * 70)
        split_stats = info.split_dataset(
            train_path=self.train_split_path,
            test_path=self.test_split_path,
            test_size=0.2,
            random_state=42,
        )

        print("\n" + "=" * 70)
        print("STEP 3: Augment TRAINING Set Only")
        print("=" * 70)
        augmentor = Augmentor(
            self.train_split_path,
            self.augmented_train_path,
            classes,
        )
        aug_stats = augmentor.perform_augmentation(target_count=1000)

        print("\n" + "=" * 70)
        print("STEP 4: Extract Features")
        print("=" * 70)
        extractor = FeatureExtractor(
            self.augmented_train_path,
            classes,
            n_jobs=-1,
        )
        X_train, X_test, y_train, y_test = extractor.extract_features_split(
            train_path=self.augmented_train_path,
            test_path=self.test_split_path,
        )

        print("\n" + "=" * 70)
        print("STEP 5: Train k-NN")
        print("=" * 70)
        knn_trainer = KNNTrainer(features_path="features/train_features.pkl")
        knn_trainer.classes = classes

        best_knn = knn_trainer.hyperparameter_tuning(X_train, y_train)
        knn_trainer.model = best_knn
        knn_accuracy, _ = knn_trainer.evaluate(X_test, y_test)
        knn_trainer.save_model("knn_model.pkl")

        print("\n" + "=" * 70)
        print("STEP 6: Train SVM")
        print("=" * 70)
        svm_trainer = SVMTrainer(features_path="features/train_features.pkl")
        svm_trainer.classes = classes

        best_svm = svm_trainer.hyperparameter_tuning(X_train, y_train)
        svm_trainer.model = best_svm
        svm_accuracy, _ = svm_trainer.evaluate(X_test, y_test)
        svm_trainer.save_model("svm_model.pkl")

        best = "SVM" if svm_accuracy > knn_accuracy else "k-NN"

        return {
            "knn_accuracy": knn_accuracy,
            "svm_accuracy": svm_accuracy,
            "best_model": best,
            "split_stats": split_stats,
            "aug_stats": aug_stats,
        }
