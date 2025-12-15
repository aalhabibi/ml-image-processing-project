import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    BASELINE Feature Extraction for Waste Classification

    Lean feature set optimized for:
    1. Fast extraction
    2. Good generalization
    3. Minimal redundancy
    4. Essential discriminative power

    Total: 193 features (baseline - essential only)
    """

    def __init__(
        self,
        dataset_path,
        classes,
        n_jobs=-1,
        save_dir="./features",
    ):
        self.dataset_path = Path(dataset_path)
        self.classes = classes
        self.scaler = StandardScaler()
        self.n_jobs = n_jobs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("BASELINE FEATURE EXTRACTOR FOR WASTE CLASSIFICATION")
        print("=" * 70)
        print("\nFeature Composition (Baseline - Essential Only):")
        print("  1. HSV Histogram            : 96 features")
        print("  2. Color Moments (HSV)      : 9 features")
        print("  3. LBP Texture              : 32 features")
        print("  4. HOG Descriptors          : 36 features")
        print("  5. Haralick Texture         : 20 features")
        print("  " + "-" * 50)
        print("  TOTAL                       : 193 features")
        print("\n  ✓ Removed: Hu moments, FFT, Gabor, edges, stats")
        print("  ✓ Lean baseline for faster training & better generalization")
        print("=" * 70 + "\n")

    # ========================================================================
    # FEATURE GROUP 1: COLOR HISTOGRAM (96 features)
    # ========================================================================
    def extract_color_histogram(self, image, bins=32):
        """Color histogram in HSV space"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        h = cv2.normalize(h, None).flatten()
        s = cv2.normalize(s, None).flatten()
        v = cv2.normalize(v, None).flatten()
        return np.concatenate([h, s, v])

    # ========================================================================
    # FEATURE GROUP 2: COLOR MOMENTS (9 features) - NEW!
    # ========================================================================
    def extract_color_moments(self, image):
        """
        Extract color moments (mean, std, skewness) for each HSV channel

        Justification:
        - Faster than histograms but highly discriminative
        - Mean: dominant color
        - Std: color variation
        - Skewness: distribution asymmetry
        - Different materials have distinct color moment signatures
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        moments = []
        for channel in cv2.split(hsv):
            mean = np.mean(channel)
            std = np.std(channel)
            # Skewness (third moment)
            skew = np.mean(((channel - mean) / (std + 1e-6)) ** 3)
            moments.extend([mean, std, skew])

        return np.array(moments)

    # ========================================================================
    # FEATURE GROUP 3: TEXTURE (32 features)
    # ========================================================================
    def extract_texture_features(self, image, P=8, R=1, bins=32):
        """LBP texture features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        return hist

    # ========================================================================
    # FEATURE GROUP 4: EDGE FEATURES (17 features)
    # ========================================================================
    def extract_edge_features(self, image, bins=16):
        """Edge density and orientation histogram"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        orientation = np.arctan2(sobely, sobelx)
        hist, _ = np.histogram(orientation, bins=bins, range=(-np.pi, np.pi))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        return np.concatenate([[edge_density], hist])

    # ========================================================================
    # FEATURE GROUP 5: STATISTICAL FEATURES (12 features)
    # ========================================================================
    def extract_statistical_features(self, image):
        """Statistical moments per BGR channel"""
        features = []
        for ch in cv2.split(image):
            features += [np.mean(ch), np.std(ch), np.min(ch), np.max(ch)]
        return np.array(features)

    # ========================================================================
    # FEATURE GROUP 4: HOG FEATURES (36 features) - BASELINE
    # ========================================================================
    def extract_hog_features(self, image):
        """
        Compact HOG features for baseline

        Settings optimized for speed and essential shape info:
        - 64x64 image (fast processing)
        - 32x32 cells (coarse but captures main structure)
        - 2x2 blocks → 1 block total
        - 9 orientations × 4 cells = 36 features

        Captures essential shape without over-complicating
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        fd = hog(
            gray,
            orientations=9,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )

        return fd

    # ========================================================================
    # FEATURE GROUP 7: SHAPE FEATURES (7 features)
    # ========================================================================
    def extract_shape_features(self, image):
        """Hu moments for shape invariance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments.flatten()

    # ========================================================================
    # FEATURE GROUP 5: HARALICK TEXTURE (20 features) - BASELINE
    # ========================================================================
    def extract_haralick_features(self, image):
        """
        Baseline Haralick texture features

        Uses distance=1 only for compact representation:
        - 4 angles × 5 properties = 20 features
        - Captures essential texture without redundancy
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = (gray / 32).astype(np.uint8)

        # Single distance for baseline
        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(
            gray,
            distances=distances,
            angles=angles,
            levels=8,
            symmetric=True,
            normed=True,
        )

        features = []
        properties = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
        ]

        for prop in properties:
            values = graycoprops(glcm, prop)
            features.extend(values.flatten())

        return np.array(features)  # 20 features

    # ========================================================================
    # FEATURE GROUP 9: ENHANCED GABOR FILTERS (8 features) - IMPROVED!
    # ========================================================================
    def extract_gabor_features(self, image):
        """
        IMPROVED: More comprehensive Gabor filters

        Enhancement:
        - Added more orientations (8 instead of 5)
        - Better coverage of texture patterns
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []

        # More comprehensive orientation coverage
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        frequencies = [0.1, 0.2]  # Two frequency scales

        for theta in orientations:
            for freq in frequencies:
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21),
                    sigma=5,
                    theta=theta,
                    lambd=10 / freq,
                    gamma=0.5,
                    psi=0,
                )
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.append(np.mean(filtered))

        return np.array(features)

    # ========================================================================
    # FEATURE GROUP 10: FREQUENCY DOMAIN FEATURES (5 features) - NEW!
    # ========================================================================
    def extract_frequency_features(self, image):
        """
        NEW: Frequency domain features using FFT

        Justification:
        - Captures periodic patterns (corrugated cardboard, labels)
        - Different materials have different frequency signatures
        - Complements spatial domain features
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # 2D FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        # Extract features from frequency spectrum
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2

        # Divide spectrum into regions
        low_freq = magnitude_spectrum[
            center_h - 10 : center_h + 10, center_w - 10 : center_w + 10
        ].mean()

        mid_freq = magnitude_spectrum[
            center_h - 30 : center_h + 30, center_w - 30 : center_w + 30
        ].mean()

        high_freq = magnitude_spectrum.mean()

        # Additional features
        freq_std = np.std(magnitude_spectrum)
        freq_energy = np.sum(magnitude_spectrum**2)

        return np.array(
            [low_freq, mid_freq, high_freq, freq_std, np.log(freq_energy + 1)]
        )

    # ========================================================================
    # MASTER FEATURE EXTRACTION - BASELINE
    # ========================================================================
    def extract_features(self, image):
        """
        Extract baseline feature vector (essential features only)

        Returns: 193-dimensional feature vector
        - HSV Histogram: 96
        - Color Moments: 9
        - LBP Texture: 32
        - HOG: 36
        - Haralick: 20

        Removed for baseline: Hu moments, FFT, Gabor, edges, stats
        """
        img = cv2.resize(image, (256, 256))

        # Extract ONLY baseline features
        color_hist = self.extract_color_histogram(img)  # 96
        color_moments = self.extract_color_moments(img)  # 9
        texture = self.extract_texture_features(img)  # 32
        hog_feat = self.extract_hog_features(img)  # 36
        haralick = self.extract_haralick_features(img)  # 20

        # Concatenate baseline features only
        feature_vector = np.concatenate(
            [
                color_hist,
                color_moments,
                texture,
                hog_feat,
                haralick,
            ]
        )

        return feature_vector

    # ========================================================================
    # DATASET PROCESSING
    # ========================================================================
    def extract_features_from_dataset(self):
        """Extract features from entire dataset"""
        print("\nExtracting features from dataset...")
        print(f"Using {self.n_jobs} parallel workers\n")

        X, y = [], []

        for class_idx, class_name in enumerate(self.classes):
            class_path = self.dataset_path / class_name
            image_files = (
                list(class_path.glob("*.jpg"))
                + list(class_path.glob("*.png"))
                + list(class_path.glob("*.jpeg"))
            )

            def process_image(img_path):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        return None
                    return self.extract_features(img)
                except Exception as e:
                    print(f"Error: {img_path}: {e}")
                    return None

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_image)(img)
                for img in tqdm(image_files, desc=f"Processing {class_name:15s}")
            )

            class_features = [r for r in results if r is not None]
            class_labels = [class_idx] * len(class_features)

            X.extend(class_features)
            y.extend(class_labels)

        X = np.array(X)
        y = np.array(y)

        print(f"\n{'='*70}")
        print("FEATURE EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Classes: {self.classes}")

        # Check for issues
        if len(X) < 600:
            print("\n⚠️  WARNING: Small dataset! Consider more augmentation.")

        # Standardize
        print("\nStandardizing features...")
        X_scaled = self.scaler.fit_transform(X)

        # NO PCA by default - it can hurt accuracy!
        # If you want dimensionality reduction, use supervised methods like LDA

        # Save
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
        print("\n🎯 Ready for classifier training!")

        return X_scaled, y

    def extract_features_split(self, train_path, test_path):
        """
        Extract features from SPLIT train/test datasets separately.

        This is the CORRECT approach:
        1. Train features are extracted from augmented training data
        2. Test features are extracted from pristine test data
        3. Scaler is fit ONLY on train data, then applied to test data

        This prevents data leakage and gives true performance metrics.
        """
        print("\n" + "=" * 70)
        print("EXTRACTING FEATURES FROM SPLIT DATASETS")
        print("=" * 70)

        # Extract TRAINING features
        print("\n[1/2] Processing TRAINING set (augmented)...")
        X_train, y_train = self._extract_from_path(train_path)

        # Fit scaler on training data ONLY
        print("\nFitting scaler on training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Extract TEST features
        print("\n[2/2] Processing TEST set (pristine)...")
        X_test, y_test = self._extract_from_path(test_path)

        # Transform test data using training scaler
        print("\nApplying training scaler to test data...")
        X_test_scaled = self.scaler.transform(X_test)

        # Save everything
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
        print("\n🎯 Ready for classifier training!")
        print("   NOTE: No train/test split needed in trainers - already split!")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _extract_from_path(self, dataset_path):
        """Helper to extract features from a specific path"""
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

            def process_image(img_path):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        return None
                    return self.extract_features(img)
                except Exception as e:
                    print(f"Error: {img_path}: {e}")
                    return None

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_image)(img)
                for img in tqdm(image_files, desc=f"  {class_name:15s}")
            )

            class_features = [r for r in results if r is not None]
            class_labels = [class_idx] * len(class_features)

            X.extend(class_features)
            y.extend(class_labels)

        return np.array(X), np.array(y)
