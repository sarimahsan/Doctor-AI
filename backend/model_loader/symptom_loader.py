import torch
import joblib
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.utils.explain import ExplainableClassifier
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access your token
hf_token = os.getenv("HF_TOKEN")

# Hugging Face repo info
HF_REPO_ID = "sarimahsan101/medical_symptom_transformer"
LABEL_ENCODER_FILE = "label_encoder.pkl"
HF_TOKEN = hf_token  # <-- add your Hugging Face token here

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ============================
# Load tokenizer & model
# ============================ 
print("[INFO] Loading tokenizer from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(
    HF_REPO_ID,
    use_fast=False,  # important!
    use_auth_token=HF_TOKEN
)
print("[SUCCESS] Tokenizer loaded")

print("[INFO] Loading model from Hugging Face Hub...")
model = AutoModelForSequenceClassification.from_pretrained(
    HF_REPO_ID,
    use_auth_token=HF_TOKEN
)
print("[SUCCESS] Model loaded")

model.to(device)
model.eval()
print("[INFO] Model moved to device and set to eval mode")

# ============================
# Load label encoder
# ============================
print("[INFO] Downloading label encoder from Hugging Face Hub...")
label_encoder_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=LABEL_ENCODER_FILE,
    token=HF_TOKEN
)

label_encoder = joblib.load(label_encoder_path)
print("[SUCCESS] Label encoder loaded")

# ============================
# Initialize explainer
# ============================
explainer = ExplainableClassifier(
    model=model,
    tokenizer=tokenizer,
    label_encoder=label_encoder,
    device=device
)

print("[SUCCESS] ExplainableClassifier initialized successfully")
