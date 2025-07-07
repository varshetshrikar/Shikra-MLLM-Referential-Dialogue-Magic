'''
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from shikra import ShikraLlamaForCausalLM
import torch


def apply_delta(base_model_path, target_model_path, delta_path):
    print("Loading base model")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
            local_files_only=True,
            trust_remote_code=True
        )

        print("Resizing base model embeddings...")
        delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, trust_remote_code=True)
        base_model.resize_token_embeddings(len(delta_tokenizer))
    except Exception as e:
        print("❌ Failed to load base model:", e)
        exit(1)

    print("Loading delta")
    delta_model = ShikraLlamaForCausalLM.from_pretrained(
        delta_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    print("Applying delta to base model")
    base_state_dict = base_model.state_dict()
    for name, param in tqdm(delta_model.state_dict().items(), desc="Applying delta"):
        if name not in base_state_dict:
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
            continue
        if param.data.shape == base_state_dict[name].shape:
            param.data += base_state_dict[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
                f'{name} shape mismatch: {param.data.shape} vs {base_state_dict[name].shape}'
            bparam = base_state_dict[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

    print("Saving merged model")
    delta_model.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)
    print("✅ Done! Saved to:", target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--delta", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base, args.target, args.delta)
'''

import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from shikra import ShikraLlamaForCausalLM
import torch
import os


def apply_delta(base_model_path, target_model_path, delta_path):
    print("🔹 Loading base and delta tokenizers...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, trust_remote_code=True)

    print("🔹 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Resize embeddings if needed
    base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    delta_vocab_size = len(delta_tokenizer)
    if base_vocab_size != delta_vocab_size:
        print(f"⚠️ Resizing base model embeddings from {base_vocab_size} to {delta_vocab_size}")
        base_model.resize_token_embeddings(delta_vocab_size)

    print("🔹 Loading delta model...")
    delta_model = ShikraLlamaForCausalLM.from_pretrained(
        delta_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("🔧 Applying delta to base model...")
    base_state_dict = base_model.state_dict()
    for name, param in tqdm(delta_model.state_dict().items(), desc="Merging weights"):
        if name not in base_state_dict:
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f"{name} not found in base model"
            continue
        if param.shape == base_state_dict[name].shape:
            param.data += base_state_dict[name].to(param.device)
        else:
            print(f"⚠️ Skipping mismatched weight: {name} — {param.shape} vs {base_state_dict[name].shape}")

    print("💾 Saving merged model...")
    os.makedirs(target_model_path, exist_ok=True)
    delta_model.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)
    print(f"✅ Done! Merged model saved at: {target_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Path to base model (e.g., LLaMA-7B)")
    parser.add_argument("--target", type=str, required=True, help="Path to save merged model")
    parser.add_argument("--delta", type=str, required=True, help="Path to delta model (e.g., Shikra delta)")
    args = parser.parse_args()

    apply_delta(args.base, args.target, args.delta)
