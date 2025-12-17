"""Run the corrected waste-classification pipeline."""

from pipeline.pipeline import WastePipeline


def main():

    pipeline = WastePipeline(
        dataset_path="data/dataset",
        train_split_path="data/train_split",
        test_split_path="data/test_split",
        augmented_train_path="data/augmented_train",
    )

    results = pipeline.run()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nResults Summary:")
    print(
        f"  k-NN Accuracy: {results['knn_accuracy']:.4f} ({results['knn_accuracy']*100:.2f}%)"
    )
    print(
        f"  SVM Accuracy:  {results['svm_accuracy']:.4f} ({results['svm_accuracy']*100:.2f}%)"
    )
    print(f"  Best Model:    {results['best_model']}")


if __name__ == "__main__":
    main()
