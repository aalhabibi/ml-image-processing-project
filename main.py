#!/usr/bin/env python3
"""
Corrected Pipeline - Prevents Data Leakage

This script runs the FIXED pipeline that:
1. Splits data BEFORE augmentation
2. Augments ONLY training data
3. Keeps test data pristine

This ensures no data leakage and gives true performance metrics.
"""

from pipeline.pipeline import WastePipeline


def main():
    print("\n" + "=" * 70)
    print("WASTE CLASSIFICATION - CORRECTED PIPELINE")
    print("=" * 70)
    print("\n🔧 This version FIXES the data leakage issue!")
    print("\nWhat changed:")
    print("  ❌ OLD: Augment entire dataset → then split (DATA LEAKAGE!)")
    print("  ✅ NEW: Split first → augment only training set")
    print("\nWhy this matters:")
    print("  - Old way: Test set had augmented copies of training images")
    print("  - New way: Test set is completely independent")
    print("  - Result: TRUE generalization performance")
    print("=" * 70)

    # Initialize pipeline with corrected paths
    pipeline = WastePipeline(
        dataset_path="data/dataset",  # Your original dataset
        train_split_path="data/train_split",  # Will store train split
        test_split_path="data/test_split",  # Will store test split
        augmented_train_path="data/augmented_train",  # Augmented train only
    )

    # Run the corrected pipeline
    results = pipeline.run()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\n📊 Results Summary:")
    print(
        f"  k-NN Accuracy: {results['knn_accuracy']:.4f} ({results['knn_accuracy']*100:.2f}%)"
    )
    print(
        f"  SVM Accuracy:  {results['svm_accuracy']:.4f} ({results['svm_accuracy']*100:.2f}%)"
    )
    print(f"  Best Model:    {results['best_model']}")

    print("\n💡 Expect to see LOWER accuracy than before!")
    print("   This is actually GOOD - it means:")
    print("   ✓ No artificial inflation from data leakage")
    print("   ✓ Real-world performance estimate")
    print("   ✓ Model will generalize better to new images")


if __name__ == "__main__":
    main()
