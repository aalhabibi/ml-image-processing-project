from train.knn.train_knn import KNNTrainer
from train.svm.train_svm import SVMTrainer
from .data_loader import DatasetInfo
from .augmentation import Augmentor
from .feature_extraction import FeatureExtractor


class WastePipeline:
    def __init__(self, dataset_path="dataset", output_path="augmented_dataset"):
        self.dataset_path = dataset_path
        self.output_path = output_path

    def run(self):
        # 1. Load dataset info
        info = DatasetInfo(self.dataset_path)
        classes, counts = info.load_dataset_info()

        # # 2. Augmentation
        # augmentor = Augmentor(self.dataset_path, self.output_path, classes)
        # augmentor.perform_augmentation(target_count=1000)

        # # 3. Feature extraction with PCA
        # extractor = FeatureExtractor(
        #     self.output_path, classes, n_jobs=-1, 
        # )
        # extractor.extract_features_from_dataset()

        # 4. Train k-NN Classifier
        print("\n" + "=" * 70)
        print("TRAINING k-NN CLASSIFIER")
        print("=" * 70)

        knn_trainer = KNNTrainer(features_path="features/processed_features.pkl")
        X_train, X_test, y_train, y_test = knn_trainer.load_data()
        best_model = knn_trainer.hyperparameter_tuning(X_train, y_train)
        knn_trainer.model = best_model
        knn_accuracy, _ = knn_trainer.evaluate(X_test, y_test)
        print(f"  k-NN Test Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
        knn_trainer.save_model("knn_model.pkl")

        # # 5. Train SVM Classifier
        # print("\n" + "=" * 70)
        # print("TRAINING SVM CLASSIFIER")
        # print("=" * 70)

        # svm_trainer = SVMTrainer(features_path="features/processed_features.pkl")
        # X_train, X_test, y_train, y_test = svm_trainer.load_data()
        # best_svm = svm_trainer.hyperparameter_tuning(X_train, y_train)
        # svm_trainer.model = best_svm
        # svm_accuracy, _ = svm_trainer.evaluate(X_test, y_test)
        # print(f"  SVM Test Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        # svm_trainer.save_model("svm_model.pkl")

        # # # Final comparison
        # print("\n" + "=" * 70)
        # print("FINAL RESULTS")
        # print("=" * 70)
        # print(f"k-NN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")
        # print(f"SVM Accuracy:  {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        # best = "SVM" if svm_accuracy > knn_accuracy else "k-NN"
        # best_acc = max(svm_accuracy, knn_accuracy)
        # print(f"\nBest Model: {best} with {best_acc:.4f} ({best_acc*100:.2f}%)")

        print("\nPipeline complete!")
