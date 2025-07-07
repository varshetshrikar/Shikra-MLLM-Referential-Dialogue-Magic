_base_ = [
    '/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra-7b',             # model architecture
    '_/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra-main/config/_base_/train/shikra_fsdp.py'         # training strategy
]

# Override just the data_args and training_args:
data_args = dict(
  train=dict(
    type='ShikraConcat',      # use a unique name
    datasets=[
      dict(type='InstructDataset',
           filename='/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra_data/pointQA_local_train.jsonl'),
      dict(type='InstructDataset',
           filename='/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra_data/CWB_flickr30k_train.jsonl'),
    ],
    probabilities=[1.0, 1.0],  # must match dataset count
  ),
    validation=None,
    test=None,
    collator_kwargs=dict(padding=True, max_length=1024),
    gen_kwargs=dict(max_new_tokens=1024, num_beams=1),
)

training_args = dict(
    num_train_epochs=3,
    max_steps=2000,                       # since InstructDataset DOES implement __len__, you can drop this if you like
    per_device_train_batch_size=8,        # set to whatever fits your GPU
    output_dir='./media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra_output',
    overwrite_output_dir=True,
)
