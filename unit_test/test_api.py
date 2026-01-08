# tests/test_api.py
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

# Import your FastAPI app
from backend.app import app
from backend.schemas.request import TextInput  # for input schema

client = TestClient(app)

# ---------------------------
# Helper function to create a dummy image
# ---------------------------
def create_dummy_image(color=(255, 0, 0)):
    """
    Create an in-memory RGB image for testing
    """
    img = Image.new("RGB", (224, 224), color=color)
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# ---------------------------
# Test the text prediction endpoint
# ---------------------------
def test_predict_text():
    payload = {
        "text": "I have a fever and cough",
        "top_k": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    
    # Check keys that exist in your API response
    assert "predicted_label" in json_data
    assert "confidence" in json_data
    assert "text" in json_data
    assert "top_predictions" in json_data
    
    # Optional: check top_predictions length
    assert len(json_data["top_predictions"]) == 3


# ---------------------------
# Test the image prediction endpoint
# ---------------------------
def test_predict_image():
    img_bytes = create_dummy_image()
    response = client.post(
        "/predict-image",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    json_data = response.json()
    
    # Check if both fields are present
    assert "prediction_index" in json_data
    assert "prediction_label" in json_data
    # Check prediction is within expected classes
    assert json_data["prediction_index"] in [0, 1, 2, 3]
    assert json_data["prediction_label"] in ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]

# ---------------------------
# Optional: test invalid file type
# ---------------------------
def test_predict_image_invalid_file():
    fake_file = BytesIO(b"not an image")
    response = client.post(
        "/predict-image",
        files={"file": ("fake.txt", fake_file, "text/plain")}
    )
    json_data = response.json()
    assert "error" in json_data
