from .single_image_dataset import SingleImageDataset
from .refcoco_dataset import RefCOCODataset  # ✅ ADD THIS LINE

def build_dataset(cfg, preprocessor=None):
    dataset_type = cfg.get("type", "").lower()

    if dataset_type == "single_image":
        return SingleImageDataset(cfg, preprocessor)
    elif dataset_type == "rec":
        return RefCOCODataset(cfg, preprocessor)  # ✅ ADD THIS LINE

    raise ValueError(f"[ERROR] Unknown dataset type: {dataset_type}")
