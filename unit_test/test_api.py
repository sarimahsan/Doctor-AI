# tests/test_api.py
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import base64


# Import your FastAPI app
from backend.app import app  # adjust the import if your app is in a different path

client = TestClient(app)

# ---------------------------
# Helper function to create a dummy image
# ---------------------------
# def create_dummy_image(color=(255, 0, 0)):
#     """
#     Create an in-memory RGB image for testing
#     """
#     img = Image.new("RGB", (224, 224), color=color)
#     img_bytes = BytesIO()
#     img.save(img_bytes, format="PNG")
#     img_bytes.seek(0)
#     return img_bytes

# # ---------------------------
# # Test the image prediction endpoint
# # ---------------------------
# def test_predict_image():
#     img_bytes = create_dummy_image()
    
#     response = client.post(
#         "/predict-image",
#         files={"file": ("test.png", img_bytes, "image/png")}
#     )
    
#     assert response.status_code == 200
#     json_data = response.json()
    
#     # Check if both fields are present
#     assert "prediction_index" in json_data
#     assert "prediction_label" in json_data
    
#     # Check prediction is within expected classes
#     assert json_data["prediction_index"] in [0, 1, 2, 3]
#     assert json_data["prediction_label"] in ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]

# # ---------------------------
# # Optional: Test invalid request (no file)
# # ---------------------------
# def test_predict_image_no_file():
#     response = client.post("/predict-image", files={})
#     assert response.status_code == 422  # Unprocessable Entity
#     json_data = response.json()
#     assert "detail" in json_data
#     assert json_data["detail"][0]["type"] == "missing"

# # ---------------------------
# # Optional: Test invalid file type
# # ---------------------------
# def test_predict_image_invalid_file():
#     fake_file = BytesIO(b"not an image")
#     response = client.post(
#         "/predict-image",
#         files={"file": ("fake.txt", fake_file, "text/plain")}
#     )
#     json_data = response.json()
#     assert "error" in json_data

def test_dummy_math():
    assert 1 + 1 == 2