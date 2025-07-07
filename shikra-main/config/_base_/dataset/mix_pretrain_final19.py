_base_ = ['mix_pretrain_final55.py']

# # --- mix_pretrain_final19.py ---
# data_args = dict(
#     train=dict(
#         type='ConcatDataset',
#         datasets=[
#             dict(
#                 type='InstructDataset',
#                 filename='/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra_data/pointQA_local_train.jsonl',
#                 # only needed if your JSON refers to images by filename:
#                 # image_folder='/media/bigdata/.../shikra_data/images'
#             ),
#             dict(
#                 type='InstructDataset',
#                 filename='/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra_data/CWB_flickr30k_train.jsonl',
#             ),
#             # add more entries here for each JSONL you want to include
#         ]
#     ),
from mllm.dataset.single_image_convsation import ConcatDataset as ShikraConcat

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

