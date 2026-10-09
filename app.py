"""Dark Pattern Detector: a small web app for the trained model.

Paste the text of a button, banner or product notice and the app says whether it
looks like a dark pattern, with the words that influenced the decision most.
An optional second tab reads text from a screenshot (needs Tesseract OCR; on Windows,
set the environment variable TESSERACT_CMD to the path of tesseract.exe if it is not on PATH).

Run the notebook first so that models/dark_pattern_lr.joblib exists, then:

    pip install gradio
    python app.py
"""

import os
from pathlib import Path

import joblib
import numpy as np

from text_utils import clean_text

MODEL_PATH = Path(__file__).parent / "models" / "dark_pattern_lr.joblib"
model = joblib.load(MODEL_PATH)
_vocab = model.named_steps["tfidf"].get_feature_names_out()
_coef = model.named_steps["clf"].coef_[0]


def explain(text, top_n=5):
    """Return (probability of dark pattern, list of (term, contribution))."""
    cleaned = clean_text(text)
    proba = float(model.predict_proba([cleaned])[0, 1])
    weights = model.named_steps["tfidf"].transform([cleaned]).toarray()[0] * _coef
    used = np.flatnonzero(weights)
    top = used[np.argsort(-np.abs(weights[used]))][:top_n]
    return proba, [(_vocab[i], float(weights[i])) for i in top]


def analyse(text):
    if not text or not text.strip():
        return "Please enter some text."
    proba, terms = explain(text)
    verdict = "⚠️ Likely a dark pattern" if proba >= 0.5 else "✅ Looks like normal interface text"
    lines = [f"### {verdict}", f"Probability of dark pattern: **{proba:.0%}**", "",
             "Words that influenced the decision most:"]
    for term, w in terms:
        direction = "towards dark pattern" if w > 0 else "towards normal text"
        lines.append(f"- `{term}`: {direction}")
    return "\n".join(lines)


def analyse_screenshot(image_path):
    if not image_path:
        return "Please upload a screenshot."
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "Screenshot analysis needs `pip install pytesseract pillow` and Tesseract OCR installed."
    if os.environ.get("TESSERACT_CMD"):              # e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe
        pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except pytesseract.TesseractNotFoundError:
        return ("Tesseract OCR was not found. Install it and, on Windows, set the environment "
                "variable TESSERACT_CMD to the path of tesseract.exe.")
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 3]
    if not lines:
        return "No readable text found in the screenshot."
    results = sorted(((explain(line)[0], line) for line in lines), reverse=True)
    flagged = [(p, line) for p, line in results if p >= 0.5]
    out = [f"Read {len(lines)} lines of text, {len(flagged)} look like dark patterns.", ""]
    out += [f"- **{p:.0%}**: {line}" for p, line in flagged[:10]]
    return "\n".join(out)


EXAMPLES = [
    "Hurry! Only 2 left in stock",
    "Offer ends in 0:14:59",
    "23 people are looking at this product right now",
    "No thanks, I don't want to save money",
    "Free returns within 30 days",
    "Add to cart",
]


def build_app():
    import gradio as gr

    with gr.Blocks(title="Dark Pattern Detector") as demo:
        gr.Markdown("# Dark Pattern Detector\nClassifies short website texts (buttons, banners, "
                    "product notices). Trained on e-commerce texts with TF-IDF + Logistic Regression.")
        with gr.Tab("Text"):
            text_in = gr.Textbox(label="Interface text", lines=3)
            text_out = gr.Markdown()
            gr.Button("Analyse", variant="primary").click(analyse, text_in, text_out)
            gr.Examples(EXAMPLES, inputs=text_in)
        with gr.Tab("Screenshot (optional)"):
            img_in = gr.Image(type="filepath", label="Screenshot")
            img_out = gr.Markdown()
            gr.Button("Analyse screenshot", variant="primary").click(analyse_screenshot, img_in, img_out)
    return demo


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
