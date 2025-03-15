from flask import Flask, request, jsonify, render_template
import os
import sys
import torch
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.helper_functions.path_resolver import DynamicPathResolver
from src.XAI import explain_bert

#############################################################

# Load Model
dpr = DynamicPathResolver(marker="README.md")
model_paths = {
    "en": os.path.join(dpr.path.models.bert.bert_english_curated.results._path, "checkpoint-2500"),
    "de": os.path.join(dpr.path.models.bert.bert_german_own.results._path, "checkpoint-730"),       # bert_german_own checkpoint-730    # bert_german_curated # checkpoint-2500
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    "en": explain_bert.load_model(model_paths["en"], device),
    "de": explain_bert.load_model(model_paths["de"], device),
}

#############################################################

# Load Explanations in language
explanations_data = {}

def load_explanations(lang="en"):
    global explanations_data
    if lang == 'de':
        explanations_json_path = dpr.path.app.static.json.german_explanations_json
    else:
        explanations_json_path = dpr.path.app.static.json.english_explanations_json
    explanations_data = explain_bert.load_explanations(explanations_json_path)

load_explanations("en")

def get_explanation_entry(email_index):
    """Return the explanation entry for the given email index, or None if invalid."""
    if 0 <= email_index < len(explanations_data):
        return explanations_data[email_index]
    return None

#############################################################

# Create App
app = Flask(__name__)

#############################################################

# Routes Index
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    subject = data.get("subject", "").strip()
    body = data.get("body", "").strip()

    if not subject and not body:
        return jsonify({"error": "Both subject and body cannot be empty."}), 400

    combined_text = f"{subject} \n{body}"

    try:
        detected_lang = detect(combined_text)
    except LangDetectException:
        detected_lang = "unknown"

    model_key = "de" if detected_lang.startswith("de") else "en"
    tokenizer, model = models[model_key]

    predicted_label, confidence = explain_bert.predict_label(combined_text, tokenizer, model, device)
    lime_html = explain_bert.explain_prediction(combined_text, tokenizer, model, device)
    ig_html = explain_bert.explain_with_ig(combined_text, tokenizer, model, device)

    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence,
        "language": detected_lang,
        "model_used": model_key,
        "lime_html": lime_html,
        "ig_html": ig_html
    })

#############################################################

# Routes Learn Phishing
@app.route('/learn_phishing')
def learn_phishing():
    return render_template('learn_phishing.html')


@app.route('/get_email', methods=['POST'])
def get_email():
    req = request.get_json()
    email_index = req.get('email_index', 0)
    entry = get_explanation_entry(email_index)
    if entry is None:
        return jsonify({'error': 'Invalid email index'}), 400

    return jsonify({
        'subject': entry.get('original_subject', ''),
        'body': entry.get('original_body', ''),
        'true_label': entry.get('label', 0),
        'pred_label': entry.get('pred_label', 'Unknown'),
        'confidence': entry.get('confidence', 0),
        'lime_html': entry.get('lime_html', '')
    })


@app.route('/get_email_label', methods=['POST'])
def get_email_label():
    req = request.get_json()
    email_index = req.get('email_index', 0)
    entry = get_explanation_entry(email_index)
    if entry is None:
        return jsonify({'error': 'Invalid email index'}), 400
    return jsonify({'label': entry.get('label', 0)})


@app.route('/get_lime_html', methods=['POST'])
def get_lime_html():
    req = request.get_json()
    email_index = req.get('email_index', 0)
    entry = get_explanation_entry(email_index)
    if entry is None:
        return jsonify({'error': 'Invalid email index'}), 400
    return jsonify({'lime_html': entry.get('lime_html', '')})


@app.route('/set_language', methods=['POST'])
def set_language():
    req = request.get_json()
    lang = req.get('lang', 'en')
    load_explanations(lang)  
    return jsonify({'status': 'language updated', 'lang': lang})

#############################################################

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
