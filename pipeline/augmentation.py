import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random


class Augmentor:
    def __init__(self, dataset_path, output_path, classes):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.classes = classes

    def augment_image(self, image, aug_type):
        """Apply a single augmentation safely."""
        if image is None or image.size == 0:
            return None

        try:
            if aug_type == "h_flip":
                return cv2.flip(image, 1)

            elif aug_type in ("rotation_15", "rotation_-15"):
                angle = 15 if aug_type == "rotation_15" else -15
                h, w = image.shape[:2]
                matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                return cv2.warpAffine(
                    image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT
                )

            elif aug_type == "brightness_inc":
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] *= 1.3
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            elif aug_type == "brightness_dec":
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] *= 0.7
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            elif aug_type == "zoom_in":
                h, w = image.shape[:2]
                crop_size = int(min(h, w) * 0.8)
                start_h = max((h - crop_size) // 2, 0)
                start_w = max((w - crop_size) // 2, 0)
                cropped = image[
                    start_h : start_h + crop_size, start_w : start_w + crop_size
                ]

                if cropped.size == 0:
                    return None

                return cv2.resize(cropped, (w, h))

            elif aug_type == "noise":
                noise = np.random.normal(0, 10, image.shape).astype(np.float32)
                img = np.clip(image.astype(np.float32) + noise, 0, 255)
                return img.astype(np.uint8)

        except Exception as e:
            print(f"[ERROR] Augmentation '{aug_type}' failed: {e}")
            return None

        return None

    def perform_augmentation(self, target_count=600):
        print("\n" + "=" * 60)
        print("STARTING DATA AUGMENTATION")
        print(f"Target: {target_count} images per class")
        print("=" * 60)

        self.output_path.mkdir(exist_ok=True)

        augmentation_types = [
            "h_flip",
            "rotation_15",
            "rotation_-15",
            "brightness_inc",
            "brightness_dec",
            "zoom_in",
        ]

        stats = {}

        for class_name in self.classes:
            print(f"\nProcessing class: {class_name}")

            class_input = self.dataset_path / class_name
            class_output = self.output_path / class_name
            class_output.mkdir(exist_ok=True)

            image_files = [
                f
                for f in class_input.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            original_count = len(image_files)

            target_aug = max(0, target_count - original_count)

            for img_file in tqdm(
                image_files, desc=f"Copying {class_name}", unit=" images"
            ):
                img = cv2.imread(str(img_file))

                if img is None:
                    print(f"[WARNING] Skipping unreadable image: {img_file}")
                    continue

                cv2.imwrite(str(class_output / img_file.name), img)

            aug_count = 0

            if original_count > 0 and target_aug > 0:
                aug_per_image = target_aug // original_count
                extra_augs = target_aug % original_count
            else:
                aug_per_image = 0
                extra_augs = 0

            print(f"  Original: {original_count}, Need: {target_aug} augmented images")
            print(f"  Strategy: {aug_per_image} augs/image + {extra_augs} extra")

            for idx, img_file in enumerate(
                tqdm(image_files, desc=f"Augmenting {class_name}")
            ):
                img = cv2.imread(str(img_file))

                if img is None:
                    print(f"[WARNING] Cannot augment unreadable image: {img_file}")
                    continue

                num_augs = aug_per_image + (1 if idx < extra_augs else 0)

                for i in range(num_augs):
                    aug_type = random.choice(augmentation_types)
                    aug_img = self.augment_image(img, aug_type)

                    # If augmentation failed — skip
                    if aug_img is None or aug_img.size == 0:
                        print(f"[WARNING] Empty augmentation: {img_file} ({aug_type})")
                        continue

                    name = f"{img_file.stem}_aug_{aug_type}_{i}{img_file.suffix}"
                    cv2.imwrite(str(class_output / name), aug_img)
                    aug_count += 1

            stats[class_name] = {
                "original": original_count,
                "augmented": aug_count,
                "final": original_count + aug_count,
                "increase_percentage": (
                    (aug_count / original_count * 100) if original_count > 0 else 0
                ),
            }

        # Save stats
        with open(self.output_path / "augmentation_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print("\nAugmentation complete!\n")
        return stats
