import os
import json

# 👉 SET THESE TO MATCH YOUR ACTUAL PATHS
jsonl_in = "/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra_data/REC_ref3_train.jsonl"   # Your original JSONL file
jsonl_out = "/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra_data/REC_ref3_train_filtered.jsonl"  # Output filtered file
image_dir = "/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/sagar/varshet/shikra-main/images/train2014"     # Folder containing COCO_train2014 images

total, kept, missing = 0, 0, 0

with open(jsonl_in, "r") as f_in, open(jsonl_out, "w") as f_out:
    for line in f_in:
        ann = json.loads(line)
        img_file = ann.get("img_path", "")
        image_path = os.path.join(image_dir, img_file)
        total += 1

        if os.path.exists(image_path):
            f_out.write(json.dumps(ann) + "\n")
            kept += 1
        else:
            missing += 1

print(f"\n✅ Done filtering.")
print(f"🔢 Total: {total}, ✅ Kept: {kept}, ❌ Missing: {missing}")
