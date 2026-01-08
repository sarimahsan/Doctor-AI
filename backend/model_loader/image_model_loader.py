# model_loader/image_model_loader.py
import torch
from huggingface_hub import hf_hub_download
from backend.utils.image_preprocessing import ImagePreprocessor

HF_REPO_ID = "sarimahsan101/xray_disease_cnn_model"
HF_FILENAME = "cnn_model_full.pt"  # use the full model

class CNNPredictor:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        self.model = model.to(self.device)
        self.model.eval()
        print("[INFO] Model set to evaluation mode")

        self.preprocessor = ImagePreprocessor()
        print("[INFO] Image preprocessor initialized")

    @classmethod
    def load_full_model(cls):
        print("[INFO] Downloading full TorchScript model...")
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        model = torch.jit.load(model_path, map_location="cpu")
        print("[SUCCESS] TorchScript model loaded successfully")
        return cls(model)

    @classmethod
    def load_state_dict(cls, model_class):
        print("[INFO] Downloading state dict...")
        state_dict_path = hf_hub_download(repo_id=HF_REPO_ID, filename="cnn_model_state_dict.pth")
        model = model_class()
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("[SUCCESS] State dict loaded successfully")
        return cls(model)

    def predict(self, image):
        tensor = self.preprocessor.preprocess(image).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            prediction = torch.argmax(outputs, dim=1).item()
        return prediction
