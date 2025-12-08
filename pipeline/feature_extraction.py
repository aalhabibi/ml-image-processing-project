import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    Enhanced Feature Extraction for Waste Classification

    Feature Vector Composition (~477 features):
    1. Color Features (96) - HSV histograms
    2. Texture Features (32) - Local Binary Patterns
    3. Edge Features (17) - Canny + orientation histogram
    4. Statistical Features (12) - Mean, std, min, max per channel
    5. HOG Features (288) - Histogram of Oriented Gradients
    6. Shape Features (7) - Hu moments for shape description
    7. Haralick Texture (20) - GLCM-based texture features
    8. Gabor Features (5) - Multi-scale texture analysis

    Total: ~477 features (scalable based on performance needs)
    """

    def __init__(
        self, dataset_path, classes, n_jobs=-1, use_pca=True, pca_variance=0.95
    ):
        self.dataset_path = Path(dataset_path)
        self.classes = classes
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance) if use_pca else None
        self.use_pca = use_pca
        self.n_jobs = n_jobs

        # Print feature breakdown
        print("\n" + "=" * 70)
        print("ENHANCED FEATURE EXTRACTOR")
        print("=" * 70)
        print("\nFeature Composition:")
        print("  1. Color Histogram (HSV)    : 96 features")
        print("  2. Texture (LBP)             : 32 features")
        print("  3. Edge Features             : 17 features")
        print("  4. Statistical Features      : 12 features")
        print("  5. HOG Descriptors           : 288 features")
        print("  6. Shape Features (Hu)       : 7 features")
        print("  7. Haralick Texture (GLCM)   : 20 features")
        print("  8. Gabor Filters             : 5 features")
        print("  " + "-" * 50)
        print("  TOTAL                        : ~477 features")
        print("=" * 70 + "\n")

    # ========================================================================
    # FEATURE GROUP 1: COLOR FEATURES (96 features)
    # ========================================================================
    def extract_color_histogram(self, image, bins=32):
        """
        Extract color histogram in HSV space

        Justification:
        - HSV is more robust to lighting changes than RGB
        - Hue captures the actual color (important for colored plastics)
        - Saturation captures color intensity
        - Value captures brightness
        - Different materials have distinctive color distributions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        h = cv2.normalize(h, None).flatten()
        s = cv2.normalize(s, None).flatten()
        v = cv2.normalize(v, None).flatten()
        return np.concatenate([h, s, v])

    # ========================================================================
    # FEATURE GROUP 2: TEXTURE FEATURES (32 features)
    # ========================================================================
    def extract_texture_features(self, image, P=8, R=1, bins=32):
        """
        Extract Local Binary Pattern (LBP) texture features

        Justification:
        - LBP captures local texture patterns
        - Excellent for distinguishing:
          * Smooth plastic surfaces
          * Rough cardboard texture
          * Glossy glass surfaces
          * Metallic textures
        - Rotation invariant with 'uniform' method
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        return hist

    # ========================================================================
    # FEATURE GROUP 3: EDGE FEATURES (17 features)
    # ========================================================================
    def extract_edge_features(self, image, bins=16):
        """
        Extract edge-based features using Canny and Sobel

        Justification:
        - Edge density varies between materials:
          * Metal cans: strong circular edges
          * Paper: irregular, soft edges
          * Glass: smooth, defined edges
        - Edge orientation helps distinguish shapes
        """
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
    # FEATURE GROUP 4: STATISTICAL FEATURES (12 features)
    # ========================================================================
    def extract_statistical_features(self, image):
        """
        Extract statistical moments from each color channel

        Justification:
        - Mean: overall brightness/color
        - Std: color variation
        - Min/Max: dynamic range
        - Different materials have different statistical profiles
        """
        features = []
        for ch in cv2.split(image):
            features += [np.mean(ch), np.std(ch), np.min(ch), np.max(ch)]
        return np.array(features)

    # ========================================================================
    # FEATURE GROUP 5: HOG FEATURES (288 features) - NEW!
    # ========================================================================
    def extract_hog_features(self, image):
        """
        Extract Histogram of Oriented Gradients (HOG)

        Justification:
        - HOG is THE standard for object recognition
        - Captures shape and appearance through gradient distribution
        - Excellent for distinguishing:
          * Bottles (cylindrical shapes)
          * Boxes (rectangular shapes)
          * Cans (circular profiles)
          * Crumpled vs flat items
        - Proven success in computer vision tasks
        - Relatively invariant to lighting

        Configuration:
        - 9 orientations: captures gradient directions
        - 16x16 pixels per cell: larger cells = fewer features
        - 2x2 cells per block: normalization for robustness
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize to smaller size to reduce HOG features dramatically
        gray = cv2.resize(gray, (128, 128))

        # HOG parameters optimized for waste classification
        fd = hog(
            gray,
            orientations=9,  # 9 gradient orientations
            pixels_per_cell=(16, 16),  # Larger cell size = fewer features
            cells_per_block=(2, 2),  # Block normalization
            block_norm="L2-Hys",  # Normalization method
            visualize=False,
            feature_vector=True,
        )

        return fd

    # ========================================================================
    # FEATURE GROUP 6: SHAPE FEATURES (7 features) - NEW!
    # ========================================================================
    def extract_shape_features(self, image):
        """
        Extract Hu Moments for shape description

        Justification:
        - Hu moments are translation, scale, and rotation invariant
        - Captures the overall shape of objects:
          * Bottles: elongated vertical shapes
          * Cans: circular/cylindrical
          * Paper: irregular shapes
          * Boxes: rectangular profiles
        - Complementary to HOG for shape analysis
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Calculate moments
        moments = cv2.moments(binary)

        # Calculate Hu moments
        hu_moments = cv2.HuMoments(moments)

        # Log transform for better scale
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return hu_moments.flatten()

    # ========================================================================
    # FEATURE GROUP 7: HARALICK TEXTURE FEATURES (20 features) - NEW!
    # ========================================================================
    def extract_haralick_features(self, image):
        """
        Extract Haralick texture features using GLCM
        (Gray-Level Co-occurrence Matrix)

        Justification:
        - Haralick features quantify texture at a higher level than LBP
        - Captures spatial relationships between pixels
        - Excellent for material classification:
          * Contrast: difference between materials
          * Homogeneity: uniformity of surface
          * Energy: texture smoothness
          * Correlation: linear dependencies
          * ASM: angular second moment
        - Widely used in medical imaging and material science
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce to 8 levels for computational efficiency
        gray = (gray / 32).astype(np.uint8)

        # Compute GLCM at 4 different angles for rotation invariance
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

        # Extract properties
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

        return np.array(features)

    # ========================================================================
    # FEATURE GROUP 8: GABOR FILTER FEATURES (5 features) - NEW!
    # ========================================================================
    def extract_gabor_features(self, image):
        """
        Extract Gabor filter responses for multi-scale texture

        Justification:
        - Gabor filters simulate human visual system
        - Captures texture at multiple scales and orientations
        - Particularly good for:
          * Periodic textures (corrugated cardboard)
          * Surface patterns (labels on bottles)
          * Fine vs coarse textures
        - Complements LBP and Haralick features
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = []

        # Multiple orientations and frequencies
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            for frequency in [0.1]:  # Can add more frequencies if needed
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21), sigma=5, theta=theta, lambd=10, gamma=0.5, psi=0
                )

                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.append(np.mean(filtered))

        # Add one more scale
        for theta in [np.pi / 8]:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21), sigma=3, theta=theta, lambd=5, gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.append(np.mean(filtered))

        return np.array(features)

    # ========================================================================
    # MASTER FEATURE EXTRACTION
    # ========================================================================
    def extract_features(self, image):
        """
        Extract complete feature vector from image

        Returns: ~477-dimensional feature vector
        """
        # Resize to standard size
        img = cv2.resize(image, (256, 256))

        # Extract all feature groups
        color_feat = self.extract_color_histogram(img)  # 96
        texture_feat = self.extract_texture_features(img)  # 32
        edge_feat = self.extract_edge_features(img)  # 17
        stat_feat = self.extract_statistical_features(img)  # 12
        hog_feat = self.extract_hog_features(img)  # 288
        shape_feat = self.extract_shape_features(img)  # 7
        haralick_feat = self.extract_haralick_features(img)  # 20
        gabor_feat = self.extract_gabor_features(img)  # 5

        # Concatenate all features
        feature_vector = np.concatenate(
            [
                color_feat,
                texture_feat,
                edge_feat,
                stat_feat,
                hog_feat,
                shape_feat,
                haralick_feat,
                gabor_feat,
            ]
        )

        return feature_vector

    # ========================================================================
    # DATASET PROCESSING
    # ========================================================================
    def extract_features_from_dataset(self):
        """
        Extract features from entire dataset with parallel processing
        """
        print("\nExtracting features from dataset...")
        print("Using parallel processing with", self.n_jobs, "workers\n")

        X, y = [], []

        for class_idx, class_name in enumerate(self.classes):
            class_path = self.dataset_path / class_name
            image_files = (
                list(class_path.glob("*.jpg"))
                + list(class_path.glob("*.png"))
                + list(class_path.glob("*.jpeg"))
            )

            # Parallel feature extraction
            def process_image(img_path):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        return None
                    return self.extract_features(img)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    return None

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_image)(img)
                for img in tqdm(image_files, desc=f"Processing {class_name:15s}")
            )

            # Filter out None results
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

        # Standardize features
        print("\nStandardizing features (zero mean, unit variance)...")
        X_scaled = self.scaler.fit_transform(X)

        # Apply PCA if enabled
        if self.use_pca:
            print(f"\nApplying PCA dimensionality reduction...")
            X_scaled = self.pca.fit_transform(X_scaled)
            print(f"Reduced from {X.shape[1]} to {X_scaled.shape[1]} features")
            print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")

        # Save
        print("\nSaving processed features...")
        with open("processed_features.pkl", "wb") as f:
            pickle.dump(
                {
                    "X": X_scaled,
                    "y": y,
                    "classes": self.classes,
                    "feature_dim": X_scaled.shape[1],
                    "original_dim": X.shape[1],
                    "pca_enabled": self.use_pca,
                },
                f,
            )
        with open("scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        if self.use_pca:
            with open("pca.pkl", "wb") as f:
                pickle.dump(self.pca, f)
            print("PCA model saved: pca.pkl")

        print("Features saved: processed_features.pkl")
        print("Scaler saved: scaler.pkl")
        print("\nReady for classifier training!")

        return X_scaled, y
