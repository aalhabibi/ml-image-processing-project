from pipeline.pipeline import WastePipeline

if __name__ == "__main__":
    pipeline = WastePipeline(dataset_path="./data/dataset", output_path="./data/augmented_dataset")
    pipeline.run()
