import torch
import numpy as np

class ExplainableClassifier:
    def __init__(self, model, tokenizer, label_encoder, device):
        """
        model: Hugging Face transformer model
        tokenizer: corresponding tokenizer
        label_encoder: sklearn LabelEncoder with classes
        device: "cuda" or "cpu"
        """
        self.model = model
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.device = device
        self.model.eval()

    def predict_with_explanation(self, text, top_k=3):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        logits = outputs.logits
        attentions = outputs.attentions

        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], k=top_k)

        # Tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Attention explanation (last layer, mean across heads)
        attention = attentions[-1][0].mean(dim=0)
        cls_attention = attention[0, 1:-1].cpu().numpy()  # exclude [CLS]/[SEP]

        # Build result
        results = {
            "text": text,
            "predicted_label": self.le.inverse_transform([top_indices[0].item()])[0],
            "confidence": float(top_probs[0].item()),
            "top_predictions": [
                {
                    "label": self.le.inverse_transform([idx.item()])[0],
                    "probability": float(prob.item())
                }
                for prob, idx in zip(top_probs, top_indices)
            ],
        }

        return results
