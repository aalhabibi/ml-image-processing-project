from pathlib import Path


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
