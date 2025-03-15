import torch
import lime.lime_text
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
import pandas as pd 

def load_model(model_folder, device):
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    model = AutoModelForSequenceClassification.from_pretrained(model_folder).to(device).eval()
    return tokenizer, model


def load_explanations(explanations_json_path):
    import json
    with open(explanations_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict_label(text, tokenizer, model, device, max_len=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted_label_idx = torch.max(probs, dim=-1)
    
    predicted_label = "Phishing" if predicted_label_idx.item() == 1 else "Legit"
    confidence_score = confidence.item()
    
    return predicted_label, confidence_score


def explain_prediction(text, tokenizer, model, device, max_len=256, num_samples=500):
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Legit", "Phishing"])
    
    def predict_proba(texts):
        inputs = tokenizer(
            texts, 
            return_tensors="pt",
            truncation=True,
            padding=True, 
            max_length=max_len
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.cpu()
            probs = torch.softmax(logits, dim=-1).numpy()

        return probs
    
    exp = explainer.explain_instance(
        text, 
        predict_proba, 
        num_features=10, 
        num_samples=num_samples
    )

    return exp.as_html()


def explain_with_ig(text, tokenizer, model, device, max_len=256, n_steps=20):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    baseline = torch.full_like(input_ids, tokenizer.pad_token_id).to(device)
    
    def model_forward(input_embeds):
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return torch.softmax(outputs.logits, dim=-1)
    
    lig = LayerIntegratedGradients(model_forward, model.bert.embeddings)
    
    attributions, _ = lig.attribute(
        inputs=model.bert.embeddings(input_ids),  
        baselines=model.bert.embeddings(baseline),  
        target=1, 
        return_convergence_delta=True,
        n_steps=n_steps 
    )
    
    attributions = attributions.sum(dim=-1).squeeze().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    attributions = np.abs(attributions)
    
    max_attr = max(attributions) if max(attributions) > 0 else 1
    highlighted_text = [
        f'<span style="background-color:rgba(255, 0, 0, {attr / max_attr:.2f})">{token}</span>'
        for token, attr in zip(tokens, attributions)
    ]
    
    return f'<div style="font-family: monospace;">{" ".join(highlighted_text)}</div>'
