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

        # 2. Augmentation
        augmentor = Augmentor(self.dataset_path, self.output_path, classes)
        augmentor.perform_augmentation()

        print("\nPipeline complete!")
