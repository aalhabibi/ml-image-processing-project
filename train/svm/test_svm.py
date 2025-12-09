#!/usr/bin/env python3
"""
quick_test.py - Quick test of trained model
Usage: python quick_test.py --image path/to/image.jpg
"""

import pickle
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipeline.feature_extraction import FeatureExtractor


def load_model_and_scaler():
    """Load trained model and scaler"""
    print("Loading model...")

    # Load model
    with open("./train/svm/svm_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    classes = model_data["classes"]
    conf_threshold = model_data.get("confidence_threshold", 0.4)
    margin_threshold = model_data.get("decision_margin_threshold", 0.5)

    # Load scaler
    with open("./features/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"✓ Model loaded. Classes: {classes}")

    return model, scaler, classes, conf_threshold, margin_threshold


def extract_features_from_image(image_path):
    """
    Extract features from a single image
    IMPORTANT: Must match the feature extractor used during training!
    """
    # Import your feature extractor

    # Initialize (dummy path, we only need the methods)
    extractor = FeatureExtractor(
        dataset_path="dummy",
        classes=["glass", "paper", "cardboard", "plastic", "metal", "trash"],
        n_jobs=1,
    )

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Extract features
    features = extractor.extract_features(image)

    return features, image


def predict(model, scaler, features, classes, conf_threshold, margin_threshold):
    """Make prediction with rejection mechanism"""
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Get predictions
    probas = model.predict_proba(features_scaled)
    max_proba = probas.max()
    prediction = model.predict(features_scaled)[0]

    # Get decision margin
    decision_values = model.decision_function(features_scaled)
    if decision_values.ndim == 1:
        margin = abs(decision_values[0])
    else:
        sorted_scores = np.sort(decision_values[0])
        margin = sorted_scores[-1] - sorted_scores[-2]

    # Check rejection criteria
    low_confidence = max_proba < conf_threshold
    close_to_boundary = margin < margin_threshold
    rejected = low_confidence or close_to_boundary

    if rejected:
        prediction = len(classes) - 1  # Unknown
    prob_dict = {}
    for i, cls in enumerate(classes):
        if i < probas.shape[1]:
            prob_dict[cls] = float(probas[0][i])

    return {
        "class_id": int(prediction),
        "class_label": classes[prediction],
        "confidence": float(max_proba),
        "margin": float(margin),
        "rejected": rejected,
        "low_confidence": low_confidence,
        "close_to_boundary": close_to_boundary,
        "all_probabilities": prob_dict,
    }


def test_single_image(image_path):
    """Test a single image"""
    print("\n" + "=" * 70)
    print("QUICK MODEL TEST")
    print("=" * 70)

    # Load model
    model, scaler, classes, conf_thresh, margin_thresh = load_model_and_scaler()

    # Extract features
    print(f"\nProcessing image: {image_path}")
    features, image = extract_features_from_image(image_path)
    print(f"✓ Extracted {len(features)} features")

    # Predict
    print("\nMaking prediction...")
    result = predict(model, scaler, features, classes, conf_thresh, margin_thresh)

    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"\n🎯 Predicted Class: {result['class_label'].upper()}")
    print(
        f"   Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)"
    )
    print(f"   Decision Margin: {result['margin']:.4f}")

    if result["rejected"]:
        print(f"\n⚠️  STATUS: REJECTED AS UNKNOWN")
        reasons = []
        if result["low_confidence"]:
            reasons.append(
                f"Low confidence ({result['confidence']:.3f} < {conf_thresh})"
            )
        if result["close_to_boundary"]:
            reasons.append(
                f"Close to boundary (margin {result['margin']:.3f} < {margin_thresh})"
            )
        print(f"   Reason: {' AND '.join(reasons)}")
    else:
        print(f"\n✅ STATUS: ACCEPTED")

    print("\n📊 All Class Probabilities:")
    sorted_probs = sorted(
        result["all_probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    for cls, prob in sorted_probs:
        bar = "█" * int(prob * 50)
        print(f"   {cls:12s}: {prob:.4f} {bar}")

    print("\n" + "=" * 70)


def test_multiple_images(image_folder):
    """Test all images in a folder"""
    print("\n" + "=" * 70)
    print("TESTING MULTIPLE IMAGES")
    print("=" * 70)

    # Load model
    model, scaler, classes, conf_thresh, margin_thresh = load_model_and_scaler()

    # Get images
    folder = Path(image_folder)
    images = (
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.jpeg"))
    )

    print(f"\nFound {len(images)} images in {image_folder}")

    results = []
    for img_path in images:
        try:
            features, _ = extract_features_from_image(img_path)
            result = predict(
                model, scaler, features, classes, conf_thresh, margin_thresh
            )
            result["path"] = img_path.name
            results.append(result)

            status = "✅" if not result["rejected"] else "⚠️"
            print(
                f"{status} {img_path.name:30s} → {result['class_label']:12s} ({result['confidence']:.3f})"
            )

        except Exception as e:
            print(f"❌ {img_path.name}: Error - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    accepted = sum(1 for r in results if not r["rejected"])
    rejected = sum(1 for r in results if r["rejected"])

    print(f"Total images: {len(results)}")
    print(f"Accepted: {accepted} ({accepted/len(results)*100:.1f}%)")
    print(f"Rejected: {rejected} ({rejected/len(results)*100:.1f}%)")

    # Class distribution
    from collections import Counter

    pred_counts = Counter([r["class_label"] for r in results])

    print("\nPredicted class distribution:")
    for cls, count in pred_counts.most_common():
        print(f"  {cls:12s}: {count} ({count/len(results)*100:.1f}%)")


def test_on_dataset(dataset_path):
    """Test on organized dataset (class folders)"""
    print("\n" + "=" * 70)
    print("TESTING ON DATASET")
    print("=" * 70)

    # Load model
    model, scaler, classes, conf_thresh, margin_thresh = load_model_and_scaler()

    dataset = Path(dataset_path)
    known_classes = classes[:-1]  # Exclude 'unknown'

    all_correct = 0
    all_total = 0
    all_rejected = 0

    for true_class in known_classes:
        class_folder = dataset / true_class

        if not class_folder.exists():
            print(f"⚠️  Skipping {true_class} (folder not found)")
            continue

        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))

        correct = 0
        rejected = 0

        print(f"\nTesting {true_class} ({len(images)} images):")

        for img_path in images:
            try:
                features, _ = extract_features_from_image(img_path)
                result = predict(
                    model, scaler, features, classes, conf_thresh, margin_thresh
                )

                if result["rejected"]:
                    rejected += 1
                elif result["class_label"] == true_class:
                    correct += 1

                all_total += 1

            except Exception as e:
                print(f"  Error on {img_path.name}: {e}")

        all_correct += correct
        all_rejected += rejected

        accuracy = correct / len(images) * 100 if images else 0
        rejection_rate = rejected / len(images) * 100 if images else 0

        print(f"  Accuracy: {correct}/{len(images)} ({accuracy:.2f}%)")
        print(f"  Rejected: {rejected}/{len(images)} ({rejection_rate:.1f}%)")

    # Overall results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    overall_accuracy = all_correct / all_total * 100 if all_total else 0
    overall_rejection = all_rejected / all_total * 100 if all_total else 0

    print(f"Total images tested: {all_total}")
    print(f"Correct predictions: {all_correct} ({overall_accuracy:.2f}%)")
    print(f"Rejected as unknown: {all_rejected} ({overall_rejection:.1f}%)")

    # Check target
    target = 85.0
    if overall_accuracy >= target:
        print(f"\n✅ TARGET ACHIEVED! {overall_accuracy:.2f}% >= {target}%")
    else:
        print(f"\n⚠️  Below target: {overall_accuracy:.2f}% < {target}%")
        print(f"   Gap: {target - overall_accuracy:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick test of trained SVM model")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--folder", type=str, help="Folder with images")
    parser.add_argument(
        "--dataset", type=str, help="Dataset folder (with class subfolders)"
    )

    args = parser.parse_args()

    try:
        if args.image:
            test_single_image(args.image)
        elif args.folder:
            test_multiple_images(args.folder)
        elif args.dataset:
            test_on_dataset(args.dataset)
        else:
            # Default: test on original dataset
            print("No arguments provided. Testing on 'dataset' folder...")
            test_on_dataset("dataset")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure these files exist:")
        print("  - ./train/svm/svm_model.pkl")
        print("  - ./features/scaler.pkl")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
