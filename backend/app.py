from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from utils.geminiint import explain_condition 
from schemas.request import TextInput
from model_loader.symptom_loader import explainer
from fastapi import APIRouter, UploadFile, File
from PIL import Image
from model_loader.image_model_loader import CNNPredictor

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:5173"],  # React dev server ports
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including POST
    allow_headers=["*"],
)
# Path to your model folder

# Prediction endpoint
@app.post("/predict")
def predict(input: TextInput):
    # Log the input in terminal
    print(f"Received input: text='{input.text}', top_k={input.top_k}")

    # Your existing prediction logic
    result = explainer.predict_with_explanation(input.text, input.top_k)
    # predicted_condition = explain_condition(result["predicted_label"])
    return result

cnn_model = CNNPredictor.load_full_model()

CLASS_MAPPING = {
    0: "COVID",
    1: "Lung Opacity",
    2: "Normal",
    3: "Viral Pneumonia"
}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Open image
        image = Image.open(file.file)
        
        # Get numeric prediction
        pred_index = cnn_model.predict(image)
        
        # Map to human-readable label
        pred_label = CLASS_MAPPING.get(pred_index, "Unknown")
        
        return {"prediction_index": pred_index, "prediction_label": pred_label}
    
    except Exception as e:
        return {"error": str(e)}
