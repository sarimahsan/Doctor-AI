from torchvision import transforms
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # normalize RGB
        ])

    def preprocess(self, image: Image.Image):
        image = image.convert("RGB")  # ensure 3 channels
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # add batch dimension
