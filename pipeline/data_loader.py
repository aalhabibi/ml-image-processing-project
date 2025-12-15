from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import random


class DatasetInfo:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.classes = []

    def load_dataset_info(self):
        """Load class names and count images per class"""
        self.classes = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        print(f"Found {len(self.classes)} classes: {self.classes}")

        image_counts = {}
        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            images = (
                list(class_path.glob("*.jpg"))
                + list(class_path.glob("*.png"))
                + list(class_path.glob("*.jpeg"))
            )
            image_counts[class_name] = len(images)

        print("\nOriginal dataset distribution:")
        for cls, count in image_counts.items():
            print(f"  {cls}: {count} images")

        return self.classes, image_counts

    def split_dataset(self, train_path, test_path, test_size=0.2, random_state=42):
        """
        Split original dataset into train/test BEFORE augmentation.

        This prevents data leakage by ensuring augmented versions of test images
        don't appear in the training set.

        Args:
            train_path: Path to save training split
            test_path: Path to save test split
            test_size: Fraction of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
        """
        print("\n" + "=" * 70)
        print("SPLITTING DATASET (BEFORE AUGMENTATION)") 
        print("=" * 70)
        print(f"Train path: {train_path}")
        print(f"Test path:  {test_path}")
        print(f"Test size:  {test_size * 100:.0f}%")
        print(f"Random seed: {random_state}")

        train_path = Path(train_path)
        test_path = Path(test_path)

        # Create directories
        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)

        split_stats = {}
        random.seed(random_state)

        for class_name in self.classes:
            # Get all images for this class
            class_path = self.dataset_path / class_name
            images = (
                list(class_path.glob("*.jpg"))
                + list(class_path.glob("*.png"))
                + list(class_path.glob("*.jpeg"))
            )

            # Sort for reproducibility
            images = sorted(images)

            # Split using sklearn (stratified by default via separate loop per class)
            train_imgs, test_imgs = train_test_split(
                images, test_size=test_size, random_state=random_state, shuffle=True
            )

            # Create class subdirectories
            (train_path / class_name).mkdir(exist_ok=True)
            (test_path / class_name).mkdir(exist_ok=True)

            # Copy files
            for img in train_imgs:
                shutil.copy2(img, train_path / class_name / img.name)

            for img in test_imgs:
                shutil.copy2(img, test_path / class_name / img.name)

            split_stats[class_name] = {
                "original": len(images),
                "train": len(train_imgs),
                "test": len(test_imgs),
            }

            print(f"\n{class_name}:")
            print(f"  Original: {len(images):4d}")
            print(
                f"  Train:    {len(train_imgs):4d} ({len(train_imgs)/len(images)*100:.1f}%)"
            )
            print(
                f"  Test:     {len(test_imgs):4d} ({len(test_imgs)/len(images)*100:.1f}%)"
            )

        # Summary
        total_original = sum(s["original"] for s in split_stats.values())
        total_train = sum(s["train"] for s in split_stats.values())
        total_test = sum(s["test"] for s in split_stats.values())

        print("\n" + "=" * 70)
        print("SPLIT COMPLETE")
        print("=" * 70)
        print(f"Total original: {total_original}")
        print(f"Total train:    {total_train} ({total_train/total_original*100:.1f}%)")
        print(f"Total test:     {total_test} ({total_test/total_original*100:.1f}%)")
        print("\n⚠️  IMPORTANT: Only the TRAIN set will be augmented!")
        print("   The TEST set remains pristine to prevent data leakage.")

        return split_stats
