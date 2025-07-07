from mllm.dataset.processors.base_text_processor import BaseTextProcessor
from mllm.dataset.processors.base_image_processor import BaseImageProcessor

PROCESSORS = {
    "BaseTextProcessor": BaseTextProcessor,
    "BaseImageProcessor": BaseImageProcessor,
}

def get_processor_class(name):
    if name not in PROCESSORS:
        raise ValueError(f"[ERROR] Processor '{name}' not found in registry.")
    return PROCESSORS[name]
