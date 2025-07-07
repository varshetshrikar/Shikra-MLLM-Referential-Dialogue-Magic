import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    
)