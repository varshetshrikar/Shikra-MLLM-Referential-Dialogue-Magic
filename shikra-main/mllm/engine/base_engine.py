# import os
# import sys
# import json
# import logging
# import warnings
# from copy import deepcopy
# from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Mapping

# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# from transformers import Seq2SeqTrainer, DataCollator, DataCollatorForSeq2Seq
# from transformers.deepspeed import is_deepspeed_zero3_enabled
# from transformers.trainer import TRAINER_STATE_NAME
# from torch.optim import AdamW  # ✅ Use PyTorch's native AdamW


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout), ],
# )


# class TrainerDifferentCollatorMixin:
#     def __init__(self,
#                  *args,
#                  train_collator: Optional[DataCollator] = None,
#                  eval_collator: Optional[DataCollator] = None,
#                  test_collator: Optional[DataCollator] = None,
#                  **kwargs):
#         if train_collator is None and eval_collator is None and test_collator is None:
#             raise ValueError("use different collator for trainer but get no collator function.")
#         if eval_collator is not None and test_collator is not None and eval_collator != test_collator:
#             warnings.warn('[WARNING!!!] use different collator for eval and test. but maybe do_eval and '
#                           'do_predict both use trainer.predict (i.e. only test_collator is used.) u should'
#                           'check your code and know exactly what u are doing.')
#         self._train_collator = train_collator
#         self._eval_collator = eval_collator if eval_collator is not None else self._train_collator
#         self._test_collator = test_collator if test_collator is not None else self._eval_collator
#         if "data_collator" in kwargs and kwargs["data_collator"] is not None:
#             warnings.warn("use different collator for trainer but get 'data_collator' argument. It will take no effect and be ignored.")
#         super().__init__(*args, **kwargs)

#     def get_train_dataloader(self) -> DataLoader:
#         old_collator = self.data_collator
#         self.data_collator = self._train_collator
#         dataloader = super().get_train_dataloader()
#         self.data_collator = old_collator
#         return dataloader

#     def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
#         old_collator = self.data_collator
#         self.data_collator = self._eval_collator
#         dataloader = super().get_eval_dataloader(eval_dataset)
#         self.data_collator = old_collator
#         return dataloader

#     def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
#         old_collator = self.data_collator
#         self.data_collator = self._test_collator
#         dataloader = super().get_test_dataloader(test_dataset)
#         self.data_collator = old_collator
#         return dataloader


# class TrainerForMMLLM(TrainerDifferentCollatorMixin, Seq2SeqTrainer):
#     def prediction_step(
#             self,
#             model: nn.Module,
#             inputs: Dict[str, Union[torch.Tensor, Any]],
#             prediction_loss_only: bool,
#             ignore_keys: Optional[List[str]] = None,
#     ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

#         if not self.args.predict_with_generate or prediction_loss_only:
#             return super().prediction_step(
#                 model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
#             )

#         has_labels = "labels" in inputs
#         inputs = self._prepare_inputs(inputs)

#         gen_kwargs = self._gen_kwargs.copy()
#         if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
#             gen_kwargs["max_length"] = self.model.config.max_length
#         gen_kwargs["num_beams"] = (
#             gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
#         )
#         default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
#         gen_kwargs["synced_gpus"] = (
#             gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
#         )

#         filter_keys = ["labels"]
#         for k in inputs:
#             if k not in filter_keys:
#                 gen_kwargs[k] = inputs[k]
#         self._logging_generate_kwargs(gen_kwargs.keys())

#         with torch.inference_mode():
#             with self.compute_loss_context_manager():
#                 generated_tokens = self.model.generate(**gen_kwargs)

#         if self.model.generation_config._from_model_config:
#             self.model.generation_config._from_model_config = False

#         generation_inputs = inputs['input_ids']
#         generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

#         gen_config = self.model.generation_config
#         if generated_tokens.shape[-1] < gen_config.max_length:
#             generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
#         elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
#             generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

#         loss = None

#         if self.args.prediction_loss_only:
#             return loss, None, None

#         if has_labels:
#             labels = inputs["labels"]
#             if labels.shape[-1] < gen_config.max_length:
#                 labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
#             elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
#                 labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
#         else:
#             labels = None

#         return loss, generated_tokens, labels

#     def _logging_generate_kwargs(self, keys):
#         if not hasattr(self, '_generate_kwargs'):
#             self._generate_kwargs = None
#         if self._generate_kwargs != keys:
#             self._generate_kwargs = keys
#             logger.warning(f"generate use kwargs: {keys}")

#     def save_prediction(self, predict_results, file_key_prefix='predict'):
#         if not self.is_world_process_zero():
#             return

#         import numpy as np
#         os.makedirs(self.args.output_dir, exist_ok=True)
#         np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_predictions.npy"), predict_results.predictions)
#         np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_label_ids.npy"), predict_results.label_ids)

#         preds, targets = predict_results.predictions, predict_results.label_ids
#         origin_preds, origin_targets = preds, targets
#         preds, targets = deepcopy(preds), deepcopy(targets)
#         logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")

#         os.makedirs(self.args.output_dir, exist_ok=True)
#         with open(os.path.join(self.args.output_dir, f'{file_key_prefix}_extra_prediction.jsonl'), 'a', encoding="utf-8") as g:
#             for p, t, pi, ti in tqdm(
#                     zip(preds, targets, origin_preds, origin_targets),
#                     total=len(preds), desc=f"saving prediction for {file_key_prefix}",
#             ):
#                 p[p < 0] = self.tokenizer.pad_token_id
#                 t[t < 0] = self.tokenizer.pad_token_id
#                 p = self.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#                 t = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#                 obj = dict(pred=p, target=t)
#                 g.write(json.dumps(obj) + '\n')
#                 g.flush()

#     def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
#         if self.fsdp is not None:
#             if output_dir is None:
#                 output_dir = self.args.output_dir
#             from torch.distributed.fsdp import (
#                 FullyShardedDataParallel as FSDP,
#                 FullStateDictConfig,
#                 StateDictType,
#             )
#             save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#             with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
#                 cpu_state_dict = self.model.state_dict()
#             if self.args.should_save:
#                 self._save(output_dir, state_dict=cpu_state_dict)
#             if self.args.push_to_hub and not _internal_call:
#                 self.push_to_hub(commit_message="Model save")
#         else:
#             super().save_model(output_dir, _internal_call)

#     def plot_loss(self) -> None:
#         if not self.is_world_process_zero():
#             return

#         training_args = self.args
#         FIGURE_NAME = "trainer_state.png"
#         import matplotlib.pyplot as plt
#         data = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME), "r"))
#         train_steps, train_losses = [], []
#         for i in range(len(data["log_history"]) - 1):
#             train_steps.append(data["log_history"][i]["step"])
#             train_losses.append(data["log_history"][i]["loss"])
#         plt.figure()
#         plt.plot(train_steps, train_losses)
#         plt.title("training loss of {}".format(training_args.output_dir))
#         plt.xlabel("step")
#         plt.ylabel("training loss")
#         plt.savefig(os.path.join(training_args.output_dir, FIGURE_NAME), format="png", transparent=True, dpi=300)
#         print("Figure saved: {}".format(os.path.join(training_args.output_dir, FIGURE_NAME)))


# class Seq2SeqDataCollator(DataCollatorForSeq2Seq):
#     def __init__(self, inference_mode: bool = False, **kwargs):
#         self.inference_mode = inference_mode
#         self.text_keys = ['input_ids', 'labels', 'attention_mask']
#         super().__init__(**kwargs)

#     # def __call__(self, features: Sequence[Dict[str, Sequence]], return_tensors=None) -> Dict[str, torch.Tensor]:
#     #     for i, feature in enumerate(features):
#     #         if 'labels' not in feature:
#     #             raise KeyError(f"[❌] Feature at index {i} is missing 'labels'. This will break training.")

#     #     text_features = [{k: feature[k] for k in self.text_keys if k in feature} for feature in features]

#     #     old_padding_side = self.tokenizer.padding_side
#     #     self.tokenizer.padding_side = 'left' if self.inference_mode else 'right'
#     #     text_features = super().__call__(text_features)
#     #     self.tokenizer.padding_side = old_padding_side

#     #     return text_features

#     def __call__(self, features: Sequence[Dict[str, Sequence]], return_tensors=None) -> Dict[str, torch.Tensor]:
#         for i, feature in enumerate(features):
#             if "labels" not in feature:
#                 if "input_ids" in feature:
#                     feature["labels"] = feature["input_ids"].copy()
#                 else:
#                     raise KeyError(f"[❌] Feature at index {i} is missing both 'labels' and 'input_ids'.")

#         text_features = [{k: feature[k] for k in self.text_keys if k in feature} for feature in features]

#         old_padding_side = self.tokenizer.padding_side
#         self.tokenizer.padding_side = 'left' if self.inference_mode else 'right'
#         text_features = super().__call__(text_features)
#         self.tokenizer.padding_side = old_padding_side

#         return text_features


# class Seq2Seq2DataCollatorWithImage(Seq2SeqDataCollator):
#     def __init__(self, preprocessor, **kwargs):
#         super().__init__(tokenizer=preprocessor['text'], **kwargs)

#     def _image_process(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         if not all('image' in feature for feature in features):
#             missing = [i for i, f in enumerate(features) if 'image' not in f]
#             print(f"[❌] Features missing 'image': {[f'dict_keys({list(features[i].keys())})' for i in missing]}")
#             raise KeyError("[ERROR] Some features are missing the 'image' key. Please ensure all training examples include image data.")

#         images = [feature['image'] for feature in features]
#         images = torch.stack(images, dim=0)
#         return {"images": images}

#     def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.Tensor]:
#         ret = super().__call__(features, return_tensors)

#         if all('image' in feature for feature in features):
#             image_outputs = self._image_process(features)
#             ret.update(image_outputs)
#         else:
#             print("[⚠️] Skipping image processing. Some features do not contain 'image'. Only text data will be used for this batch.")

#         return ret


import os
import sys
import json
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Seq2SeqTrainer, DataCollator, DataCollatorForSeq2Seq
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINER_STATE_NAME
from torch.optim import AdamW  # Use PyTorch's native AdamW
from transformers import Trainer  # Add this line
from mllm.dataset.collator import get_collator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class TrainerDifferentCollatorMixin:
    def __init__(self, *args,
                 train_collator: Optional[DataCollator] = None,
                 eval_collator: Optional[DataCollator] = None,
                 test_collator: Optional[DataCollator] = None,
                 **kwargs):
        if train_collator is None and eval_collator is None and test_collator is None:
            raise ValueError("use different collator for trainer but get no collator function.")
        if eval_collator is not None and test_collator is not None and eval_collator != test_collator:
            warnings.warn("[WARNING!!!] Eval and test collators differ. Check your implementation.")
        self._train_collator = train_collator
        self._eval_collator = eval_collator if eval_collator is not None else self._train_collator
        self._test_collator = test_collator if test_collator is not None else self._eval_collator
        if "data_collator" in kwargs and kwargs["data_collator"] is not None:
            warnings.warn("'data_collator' argument will be ignored due to different collator usage.")
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._train_collator
        dataloader = super().get_train_dataloader()
        self.data_collator = old_collator
        return dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._eval_collator
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = old_collator
        return dataloader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._test_collator
        dataloader = super().get_test_dataloader(test_dataset)
        self.data_collator = old_collator
        return dataloader


class TrainerForMMLLM(TrainerDifferentCollatorMixin, Seq2SeqTrainer):
    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                        prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = gen_kwargs.get("num_beams", self.model.config.num_beams)
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", is_deepspeed_zero3_enabled())

        for k in inputs:
            if k not in ["labels"]:
                gen_kwargs[k] = inputs[k]
        self._logging_generate_kwargs(gen_kwargs.keys())

        with torch.inference_mode():
            with self.compute_loss_context_manager():
                generated_tokens = self.model.generate(**gen_kwargs)

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        generation_inputs = inputs['input_ids']
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

        if generated_tokens.shape[-1] < self.model.generation_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.model.generation_config.max_length)

        loss = None

        if prediction_loss_only:
            return loss, None, None

        labels = inputs.get("labels")
        if labels is not None and labels.shape[-1] < self.model.generation_config.max_length:
            labels = self._pad_tensors_to_max_len(labels, self.model.generation_config.max_length)

        return loss, generated_tokens, labels

    def _logging_generate_kwargs(self, keys):
        if getattr(self, '_generate_kwargs', None) != keys:
            self._generate_kwargs = keys
            logger.warning(f"generate use kwargs: {keys}")

    def save_prediction(self, predict_results, file_key_prefix='predict'):
        if not self.is_world_process_zero():
            return
        import numpy as np
        os.makedirs(self.args.output_dir, exist_ok=True)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_predictions.npy"), predict_results.predictions)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_label_ids.npy"), predict_results.label_ids)
        preds, targets = deepcopy(predict_results.predictions), deepcopy(predict_results.label_ids)
        logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")
        with open(os.path.join(self.args.output_dir, f'{file_key_prefix}_extra_prediction.jsonl'), 'a', encoding="utf-8") as g:
            for p, t in tqdm(zip(preds, targets), total=len(preds), desc=f"saving prediction for {file_key_prefix}"):
                p[p < 0] = self.tokenizer.pad_token_id
                t[t < 0] = self.tokenizer.pad_token_id
                g.write(json.dumps({"pred": self.tokenizer.decode(p, skip_special_tokens=True), "target": self.tokenizer.decode(t, skip_special_tokens=True)}) + '\n')

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if self.fsdp is not None:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = self.model.state_dict()
            if self.args.should_save:
                self._save(output_dir or self.args.output_dir, state_dict=cpu_state_dict)
        else:
            super().save_model(output_dir, _internal_call)

    def plot_loss(self):
        if not self.is_world_process_zero():
            return
        import matplotlib.pyplot as plt
        with open(os.path.join(self.args.output_dir, TRAINER_STATE_NAME), "r") as f:
            data = json.load(f)
        steps = [log["step"] for log in data["log_history"] if "loss" in log]
        losses = [log["loss"] for log in data["log_history"] if "loss" in log]
        plt.plot(steps, losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.args.output_dir, "trainer_state.png"), dpi=300)


from typing import Dict, Sequence
from transformers import DataCollatorForSeq2Seq

class Seq2SeqDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, inference_mode: bool = False, **kwargs):
        self.inference_mode = inference_mode
        self.text_keys = ['input_ids', 'labels', 'attention_mask']
        super().__init__(**kwargs)

    def __call__(self, features, return_tensors=None):
        filtered = [f for f in features if f.get("input_ids") is not None and len(f["input_ids"]) > 0]

        if len(filtered) == 0:
            raise ValueError("[❌] Data collator received an empty batch or all items missing 'input_ids'.")

        return super().__call__(filtered, return_tensors=return_tensors)






        text_features = [{k: feature[k] for k in self.text_keys if k in feature} for feature in features]
        labels = [feature.get("labels") for feature in features]

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left' if self.inference_mode else 'right'
        text_features = super().__call__(text_features, return_tensors=return_tensors)
        self.tokenizer.padding_side = old_padding_side

        if labels[0] is not None:
            text_features["labels"] = torch.stack(labels)

        return text_features


class ShikraTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        from mllm.dataset.collator import get_collator  # ensure this is imported

        cfg = kwargs.get("cfg", None)  # your YAML config is passed through here
        tokenizer = kwargs.get("tokenizer", None)
        model = kwargs.get("model", None)
        preprocessor = kwargs.get("preprocessor", None)

        if "data_collator" not in kwargs or kwargs["data_collator"] is None:
            if cfg is None:
                raise ValueError("Config not found for collator initialization.")
            kwargs["data_collator"] = get_collator(
                cfg.dataset_cfg.get("collator", None),
                tokenizer=tokenizer,
                model=model,
                preprocessor=preprocessor
            )

        super().__init__(*args, **kwargs)





    def _image_process(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not all('image' in feature for feature in features):
            missing = [i for i, f in enumerate(features) if 'image' not in f]
            raise KeyError(f"[❌] Missing 'image' key in features: {missing}")
        images = torch.stack([feature['image'] for feature in features], dim=0)
        return {"images": images}

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.Tensor]:
        ret = super().__call__(features, return_tensors)
        if all('image' in feature for feature in features):
            ret.update(self._image_process(features))
        else:
            print("[⚠️] Skipping image processing. Some features do not contain 'image'.")
        return ret

class Seq2Seq2DataCollatorWithImage:
    def __init__(self, tokenizer, model=None, preprocessor=None):
        self.tokenizer = tokenizer
        self.model = model
        self.preprocessor = preprocessor

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        pixel_values = [item.get("pixel_values") for item in batch]
        labels = [item.get("labels") for item in batch]

        batch_dict = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            ) if labels[0] is not None else None,
        }

        if pixel_values[0] is not None:
            batch_dict["pixel_values"] = torch.stack(pixel_values)

        return batch_dict
