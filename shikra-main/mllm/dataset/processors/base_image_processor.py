class BaseImageProcessor:
    def __init__(self, params=None):
        self.params = params or {}

    def __call__(self, image):
        return image  # or preprocess image here
