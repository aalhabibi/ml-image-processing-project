import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler

# Optional imports for CNN-based feature extraction
try:
    import torch
    from PIL import Image
    from torchvision import models
    from torchvision.models import ResNet50_Weights

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


class FeatureExtractor:
    """CNN feature extractor for the waste classifier."""

    def __init__(
        self,
        dataset_path,
        classes,
        n_jobs=1,
        save_dir="./features",
        feature_type="cnn",
        cnn_model_name="resnet50",
    ):
        self.dataset_path = Path(dataset_path)
        self.classes = classes
        self.scaler = StandardScaler()
        self.n_jobs = 1 if n_jobs == -1 else n_jobs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.feature_type = feature_type.lower()
        self.cnn_model_name = cnn_model_name.lower()

        if self.feature_type != "cnn":
            raise ValueError("Only feature_type='cnn' is supported now.")

        if not _TORCH_AVAILABLE:
            raise ImportError(
                "CNN feature extraction requires torch, torchvision, and Pillow. "
                "Please install them and retry."
            )

        # CNN models are not process-parallel friendly; force single worker
        self.n_jobs = 1
        self._init_cnn()
        print(
            f"Initialized feature extractor (CNN: {self.cnn_model_name.upper()}, 2048-d).\n"
        )

    def _init_cnn(self):
        """Initialize a pre-trained CNN for feature extraction."""
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.cnn_model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            # Replace final FC with identity to get penultimate features (2048-d)
            model.fc = torch.nn.Identity()
            self.cnn_feature_dim = 2048
            self.cnn_transform = weights.transforms()
        else:
            raise ValueError(
                f"Unsupported cnn_model_name: {self.cnn_model_name}. Supported: 'resnet50'"
            )

        model.eval()
        model.to(self.device)
        self.cnn_model = model

    def extract_features(self, image):
        """Extract feature vector using the CNN."""
        return self.extract_cnn_features(image)

    def extract_cnn_features(self, image):
        """Extract deep features from a pre-trained CNN (ResNet50)."""
        # Convert BGR (OpenCV) to RGB and to PIL Image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Preprocess per model's recommended transforms
        input_tensor = self.cnn_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.cnn_model(input_tensor)
        # Move to CPU and flatten
        return feats.detach().cpu().numpy().reshape(-1)

    def extract_features_from_dataset(self):
        """Extract features from the entire dataset."""
        print("\nExtracting features from dataset...")
        X, y = [], []

        for class_idx, class_name in enumerate(self.classes):
            class_path = self.dataset_path / class_name
            image_files = (
                list(class_path.glob("*.jpg"))
                + list(class_path.glob("*.png"))
                + list(class_path.glob("*.jpeg"))
            )

            for img_path in tqdm(image_files, desc=f"Processing {class_name:15s}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    feat = self.extract_features(img)
                    X.append(feat)
                    y.append(class_idx)
                except Exception as e:
                    print(f"Error: {img_path}: {e}")

        X = np.array(X)
        y = np.array(y)

        print(f"\n{'='*70}")
        print("FEATURE EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Classes: {self.classes}")

        if len(X) < 600:
            print("\n⚠️  WARNING: Small dataset! Consider more augmentation.")

        print("\nStandardizing features...")
        X_scaled = self.scaler.fit_transform(X)

        print("\nSaving...")
        with open(self.save_dir / "processed_features.pkl", "wb") as f:
            pickle.dump(
                {
                    "X": X_scaled,
                    "y": y,
                    "classes": self.classes,
                    "feature_dim": X_scaled.shape[1],
                },
                f,
            )

        with open(self.save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"✓ Features saved: {self.save_dir / 'processed_features.pkl'}")
        print(f"✓ Scaler saved: {self.save_dir / 'scaler.pkl'}")

        return X_scaled, y

    def extract_features_split(self, train_path, test_path):
        """Extract features for train/test splits without leakage."""
        print("\n" + "=" * 70)
        print("EXTRACTING FEATURES FROM SPLIT DATASETS")
        print("=" * 70)

        print("\n[1/2] Processing TRAINING set (augmented)...")
        X_train, y_train = self._extract_from_path(train_path)

        print("\nFitting scaler on training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        print("\n[2/2] Processing TEST set (pristine)...")
        X_test, y_test = self._extract_from_path(test_path)

        print("\nApplying training scaler to test data...")
        X_test_scaled = self.scaler.transform(X_test)

        print("\nSaving features and scaler...")

        with open(self.save_dir / "train_features.pkl", "wb") as f:
            pickle.dump(
                {
                    "X": X_train_scaled,
                    "y": y_train,
                    "classes": self.classes,
                    "feature_dim": X_train_scaled.shape[1],
                },
                f,
            )

        with open(self.save_dir / "test_features.pkl", "wb") as f:
            pickle.dump(
                {
                    "X": X_test_scaled,
                    "y": y_test,
                    "classes": self.classes,
                    "feature_dim": X_test_scaled.shape[1],
                },
                f,
            )

        with open(self.save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        print("\n" + "=" * 70)
        print("FEATURE EXTRACTION COMPLETE (SPLIT MODE)")
        print("=" * 70)
        print(f"Training samples: {len(X_train_scaled)} (from augmented data)")
        print(f"Test samples:     {len(X_test_scaled)} (pristine, no augmentation)")
        print(f"Feature dimension: {X_train_scaled.shape[1]}")
        print(f"\n✓ Train features: {self.save_dir / 'train_features.pkl'}")
        print(f"✓ Test features:  {self.save_dir / 'test_features.pkl'}")
        print(f"✓ Scaler saved:   {self.save_dir / 'scaler.pkl'}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _extract_from_path(self, dataset_path):
        """Extract features from a specific dataset path."""
        dataset_path = Path(dataset_path)
        X, y = [], []

        for class_idx, class_name in enumerate(self.classes):
            class_path = dataset_path / class_name

            if not class_path.exists():
                print(f"Warning: {class_path} not found, skipping...")
                continue

            image_files = (
                list(class_path.glob("*.jpg"))
                + list(class_path.glob("*.png"))
                + list(class_path.glob("*.jpeg"))
            )

            for img_path in tqdm(image_files, desc=f"  {class_name:15s}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    feat = self.extract_features(img)
                    X.append(feat)
                    y.append(class_idx)
                except Exception as e:
                    print(f"Error: {img_path}: {e}")

        return np.array(X), np.array(y)
