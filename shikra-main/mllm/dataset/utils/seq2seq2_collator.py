import torch
from torch.nn.utils.rnn import pad_sequence

class Seq2Seq2DataCollatorWithImage:
    def __init__(self, tokenizer, model=None, preprocessor=None):
        self.tokenizer = tokenizer
        self.model = model
        self.preprocessor = preprocessor

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item.get("labels") for item in batch]
        pixel_values = [item.get("pixel_values") for item in batch if "pixel_values" in item]

        batch_dict = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100) if labels[0] is not None else None,
        }

        if pixel_values:
            batch_dict["pixel_values"] = torch.stack(pixel_values)

        return batch_dict
