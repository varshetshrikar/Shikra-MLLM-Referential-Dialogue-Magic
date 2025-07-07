from .datasets import build_dataset
from torch.utils.data import DataLoader

def build_dataloader(cfg, preprocessor):
    train_dataset = build_dataset(cfg.dataset_cfg.train, preprocessor) if hasattr(cfg.dataset_cfg, 'train') else None
    val_dataset = build_dataset(cfg.dataset_cfg.val, preprocessor) if hasattr(cfg.dataset_cfg, 'val') else None

    train_loader = None
    val_loader = None

    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.run_cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.run_cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.run_cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.run_cfg.num_workers,
            pin_memory=True
        )

    return train_loader, val_loader
