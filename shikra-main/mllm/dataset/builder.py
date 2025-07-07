import os
from typing import Tuple, Type, Dict  # ✅ Required for type hints

from torch.utils.data import DataLoader
from transformers import DataCollator  # ✅ Required if using DataCollator in return type

from .refcoco_dataset import RefCOCODataset
from mllm.engine.base_engine import TrainerForMMLLM  # ✅ Required Trainer class


# Register datasets by type name
DATASETS = {
    "rec": RefCOCODataset,  # ✅ matches lowercase "rec" from YAML
}


def build_dataset(cfg, preprocessor):
    """
    Build dataset from config.

    Args:
        cfg (dict): Must include 'type' and dataset paths.
        preprocessor (dict): Dictionary with 'image' and 'text' processors.

    Returns:
        Dataset: The corresponding dataset instance.
    """
    dataset_type = cfg.get("type")
    assert dataset_type in DATASETS, f"[ERROR] Unknown dataset type '{dataset_type}'"
    dataset_class = DATASETS[dataset_type]
    return dataset_class(cfg, preprocessor)


def build_dataloader(dataset_cfg, training_args, preprocessor):
    """
    Builds train and validation dataloaders from dataset config.

    Args:
        dataset_cfg: Dictionary with "train" and "eval" dataset settings.
        training_args: HuggingFace TrainingArguments object.
        preprocessor: Dictionary with 'image' and 'text' processors.

    Returns:
        Tuple[train_loader, val_loader, tokenizer]
    """
    train_dataset = build_dataset(dataset_cfg["train"], preprocessor) if "train" in dataset_cfg else None
    val_dataset = build_dataset(dataset_cfg["eval"], preprocessor) if "eval" in dataset_cfg else None

    train_loader = None
    val_loader = None

    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
        )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, train_dataset.tokenizer if train_dataset else None


def prepare_data(*args, **kwargs):
    pass
