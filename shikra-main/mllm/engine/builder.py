from functools import partial
from typing import Tuple, Dict, Any, Type

from transformers.trainer import DataCollator

from mllm.dataset.collator import Seq2Seq2DataCollatorWithImage  # ✅ your actual collator
from .shikra import ShikraTrainer
from .base_engine import TrainerForMMLLM  # base class (used for type hinting only)

# Supported trainer types
TYPE2TRAINER = {
    'shikra': ShikraTrainer,
}

def prepare_trainer_collator(
    model_args,
    preprocessor: Dict[str, Any],
    collator_kwargs: Dict[str, Any]
) -> Tuple[Type[TrainerForMMLLM], DataCollator]:
    """
    Prepares the trainer class and its data collator.

    Args:
        model_args: arguments specifying the model type
        preprocessor: dictionary of preprocessor configurations
        collator_kwargs: kwargs to be passed into collator (e.g., tokenizer, model)

    Returns:
        trainer_cls: Trainer class (e.g., ShikraTrainer)
        data_collator: instance of DataCollator
    """
    type_ = model_args.type
    if type_ not in TYPE2TRAINER:
        raise ValueError(f"Unknown trainer type: {type_}")

    trainer_cls = TYPE2TRAINER[type_]

    # Create the collator instance
    data_collator = Seq2Seq2DataCollatorWithImage(
        tokenizer=collator_kwargs.get("tokenizer"),
        model=collator_kwargs.get("model"),
        preprocessor=preprocessor,
    )

    return trainer_cls, data_collator
