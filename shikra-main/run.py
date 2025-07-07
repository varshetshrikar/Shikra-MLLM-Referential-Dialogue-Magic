import os
import argparse
import gc
import torch
from accelerate import Accelerator
from transformers import TrainingArguments

from mllm.config.config import Config
from mllm.models.builder import build_model
from mllm.dataset.builder import build_dataloader
from mllm.dataset import registry

from mllm.engine.builder import prepare_trainer_collator  # ✅ correct import


def parse_args():
    parser = argparse.ArgumentParser(description="Run Shikra Training or Evaluation")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration YAML")
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    local_rank = accelerator.local_process_index

    print("Visible devices:", torch.cuda.device_count())
    print("Device names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print(f"[RANK {local_rank}] Using device: {device}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    # Load config
    cfg = Config.fromfile(args.cfg_path)
    run_cfg = dict(cfg.run_cfg) if hasattr(cfg, "run_cfg") else dict(cfg.run)

    # Set defaults
    run_cfg.setdefault("output_dir", "output/default")
    run_cfg.setdefault("logging_dir", os.path.join(run_cfg["output_dir"], "logs"))
    run_cfg.setdefault("report_to", [])
    run_cfg.setdefault("evaluation_strategy", "steps")
    run_cfg.setdefault("save_strategy", "epoch")
    run_cfg.setdefault("logging_steps", 10)
    run_cfg.setdefault("skip_memory_metrics", False)
    run_cfg.setdefault("full_determinism", False)

    # Convert to HF args
    run_cfg["per_device_train_batch_size"] = run_cfg.pop("train_batch_size", 1)
    run_cfg["per_device_eval_batch_size"] = run_cfg.pop("eval_batch_size", 1)
    run_cfg["learning_rate"] = float(run_cfg.pop("lr", 5e-5))
    run_cfg["weight_decay"] = float(run_cfg.get("weight_decay", 0.0))
    run_cfg["gradient_accumulation_steps"] = run_cfg.pop("accum_grad_iters", 1)
    run_cfg.pop("device_id", None)
    run_cfg["dataloader_num_workers"] = run_cfg.get("num_workers", 4)

    # Filter only allowed args
    dummy_args = TrainingArguments(output_dir="dummy")
    valid_keys = dummy_args.to_dict().keys()
    filtered_run_cfg = {k: v for k, v in run_cfg.items() if k in valid_keys}
    training_args = TrainingArguments(**filtered_run_cfg)

    # Build model
    model, _ = build_model(cfg.model_cfg, training_args)


    # Build preprocessors
    text_processor_cfg = cfg.dataset_cfg.processors.text_processor
    image_processor_cfg = cfg.dataset_cfg.processors.image_processor

    text_processor = registry.get_processor_class(text_processor_cfg.type)(text_processor_cfg.params)
    image_processor = registry.get_processor_class(image_processor_cfg.type)(image_processor_cfg.params)

    preprocessor = {
        "text": text_processor,
        "image": image_processor,
    }

    # Build dataloaders
    train_dataloader, eval_dataloader, tokenizer = build_dataloader(
        cfg["dataset_cfg"], training_args, preprocessor
    )

    # ✅ Fix: Use dictionary output from prepare_trainer_collator
    trainer_cls, data_collator = prepare_trainer_collator(
        model_args=cfg.model_cfg,
        preprocessor=preprocessor,
        collator_kwargs={
            "tokenizer": tokenizer,
            "model": model
        }
    )

    trainer = trainer_cls(
    model=model,
    args=training_args,
    train_dataset=train_dataloader.dataset if train_dataloader else None,
    eval_dataset=eval_dataloader.dataset if eval_dataloader else None,
    tokenizer=tokenizer,
    train_collator=data_collator,
    eval_collator=data_collator,
)



    if training_args.do_train:
        trainer.train()

    if training_args.do_eval:
        trainer.evaluate()

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
