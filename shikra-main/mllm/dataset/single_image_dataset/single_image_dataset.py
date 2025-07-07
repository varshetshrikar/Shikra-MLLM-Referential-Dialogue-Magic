# mllm/dataset/single_image_dataset/single_image_dataset.py

class SingleImageDataset:
    def __init__(self, cfg, preprocessor):
        self.cfg = cfg
        self.preprocessor = preprocessor

    def __getitem__(self, idx):
        # return a dummy item
        return {"image": None, "text": "placeholder"}

    def __len__(self):
        return 1
