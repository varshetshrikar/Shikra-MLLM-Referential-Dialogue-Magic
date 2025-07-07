class BaseTextProcessor:
    def __init__(self, params=None):
        self.params = params or {}

    def __call__(self, text):
        return text  # or preprocess text here
