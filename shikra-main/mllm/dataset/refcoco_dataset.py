import os
import json
from torch.utils.data import Dataset
from PIL import Image

class RefCOCODataset(Dataset):
    def __init__(self, cfg, preprocessor):
        self.ann_path = cfg.get("ann_path")
        self.image_folder = cfg.get("image_folder")
        self.image_processor = preprocessor["image"]
        self.tokenizer = preprocessor["text"]

        assert os.path.exists(self.ann_path), f"[ERROR] Annotation file not found: {self.ann_path}"
        assert os.path.exists(self.image_folder), f"[ERROR] Image folder not found: {self.image_folder}"

        with open(self.ann_path, "r") as f:
            self.annotations = [json.loads(line.strip()) for line in f if line.strip()]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # ✅ Validate image path
        img_name = ann.get("img_path", None)
        if img_name is None:
            print(f"[❌] Missing 'img_path' key at index {idx}: {ann}")
            raise KeyError(f"Missing 'img_path' key at index {idx}")

        image_path = os.path.join(self.image_folder, img_name)
        if not os.path.exists(image_path):
            print(f"[❌] Image file not found at path: {image_path}")
            raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

        # ✅ Load and process image
        image = Image.open(image_path).convert("RGB")
        processed_image = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        # ✅ Get question and answer
        question = ann.get("expression", "")
        answer = ann.get("answer", "")

        # ✅ Tokenize question
        tokenized = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        # ✅ Tokenize answer as labels for loss computation
        label_ids = self.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )["input_ids"][0]

        # ✅ Final return dict
        sample = {
            "image": processed_image,
            "question": question,
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": label_ids,  # ✅ Required for loss computation
            "answer": answer
        }

        # Optional: Debug print
        print(f"[✅] Sample {idx} returned with keys: {list(sample.keys())}")

        return sample
