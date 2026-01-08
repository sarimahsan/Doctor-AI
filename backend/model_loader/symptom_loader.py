import os
import torch
import joblib
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.explain import ExplainableClassifier
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

HF_REPO_ID = "sarimahsan101/medical_symptom_transformer"
LABEL_ENCODER_FILE = "label_encoder.pkl"

# internal cache for loaded objects
_PIPELINE_CACHE = {}

def load_symptom_pipeline(hf_token: str | None = None, device: str | None = None):
    """
    Load and return tokenizer, model, label_encoder and explainer.
    Returns a dict with keys: tokenizer, model, label_encoder, explainer, device.
    """
    if "pipeline" in _PIPELINE_CACHE:
        return _PIPELINE_CACHE["pipeline"]

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Using device: {device}")

    if hf_token is None:
        print("[WARN] HF_TOKEN not found in environment. Attempting anonymous access (may fail if repo is private).")

    # Load tokenizer & model
    try:
        print("[INFO] Loading tokenizer from Hugging Face Hub...")
        tokenizer = AutoTokenizer.from_pretrained(
            HF_REPO_ID,
            use_fast=False,
            use_auth_token=hf_token
        )
        print("[SUCCESS] Tokenizer loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {HF_REPO_ID}: {e}")

    try:
        print("[INFO] Loading model from Hugging Face Hub...")
        model = AutoModelForSequenceClassification.from_pretrained(
            HF_REPO_ID,
            use_auth_token=hf_token
        )
        print("[SUCCESS] Model loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {HF_REPO_ID}: {e}")

    model.to(device)
    model.eval()
    print("[INFO] Model moved to device and set to eval mode")

    # Load label encoder
    try:
        print("[INFO] Downloading label encoder from Hugging Face Hub...")
        label_encoder_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=LABEL_ENCODER_FILE,
            token=hf_token
        )
        label_encoder = joblib.load(label_encoder_path)
        print("[SUCCESS] Label encoder loaded")
    except Exception as e:
        raise RuntimeError(f"Failed to download or load label encoder '{LABEL_ENCODER_FILE}': {e}")

    # Initialize explainer
    try:
        explainer = ExplainableClassifier(
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            device=device
        )
        print("[SUCCESS] ExplainableClassifier initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ExplainableClassifier: {e}")

    pipeline = {
        "tokenizer": tokenizer,
        "model": model,
        "label_encoder": label_encoder,
        "explainer": explainer,
        "device": device,
    }

    _PIPELINE_CACHE["pipeline"] = pipeline
    return pipeline


def get_symptom_pipeline():
    """
    Convenience getter that reads HF_TOKEN from env and returns cached pipeline.
    """
    return load_symptom_pipeline()
