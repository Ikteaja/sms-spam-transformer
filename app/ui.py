"""
Gradio UI — SMS Spam Classifier
Mounted onto the FastAPI app at /ui  (single port, single process).

Usage (standalone):
    python app/ui.py

Usage (via FastAPI — see main.py):
    uvicorn app.main:app --reload
    → open http://localhost:8000/ui
"""

import os
from pathlib import Path

import gradio as gr
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# ---------------------------------------------------------------------------
# Model loading (shared with main.py when mounted)
# ---------------------------------------------------------------------------
_DEFAULT_DIRS = [Path("models/frozen"), Path("models/best")]
_FALLBACK_HF = "distilbert-base-uncased"  # downloaded at runtime if no fine-tuned model

_model: DistilBertForSequenceClassification = None
_tokenizer: DistilBertTokenizerFast = None
_model_label: str = ""


def _load():
    global _model, _tokenizer, _model_label
    if _model is not None:
        return

    env_path = Path(os.getenv("MODEL_DIR", ""))
    candidates = ([env_path] if env_path.name else []) + _DEFAULT_DIRS

    chosen = next((p for p in candidates if p.exists()), None)
    if chosen:
        _model_label = f"Fine-tuned · {chosen}"
        checkpoint = str(chosen)
    else:
        _model_label = f"Base DistilBERT · {_FALLBACK_HF} (no fine-tuned model found)"
        checkpoint = _FALLBACK_HF

    _tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)
    _model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    _model.eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def classify(message: str):
    _load()
    if not message.strip():
        return "⚠️ Please enter a message.", "", ""

    enc = _tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    with torch.no_grad():
        logits = _model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    spam_prob = float(probs[1])
    ham_prob = float(probs[0])
    is_spam = spam_prob > ham_prob

    # Result badge
    if is_spam:
        badge = f"🚨  SPAM  ({spam_prob:.1%} confidence)"
    else:
        badge = f"✅  HAM — Legitimate  ({ham_prob:.1%} confidence)"

    # Confidence breakdown
    bar_spam = "█" * int(spam_prob * 30) + "░" * (30 - int(spam_prob * 30))
    bar_ham = "█" * int(ham_prob * 30) + "░" * (30 - int(ham_prob * 30))
    breakdown = f"SPAM  {bar_spam}  {spam_prob:.1%}\n" f"HAM   {bar_ham}  {ham_prob:.1%}"

    explanation = _explain(message, is_spam, spam_prob)
    return badge, breakdown, explanation


def _explain(text: str, is_spam: bool, confidence: float) -> str:
    import re

    signals = []
    if re.search(r"http|www|\.com", text, re.I):
        signals.append("contains a URL")
    if re.search(r"free|win|prize|cash|award", text, re.I):
        signals.append("uses prize/money language")
    if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.25:
        signals.append("heavy use of CAPITALS")
    if text.count("!") > 1:
        signals.append(f"{text.count('!')} exclamation marks")
    if re.search(r"[£$€]", text):
        signals.append("contains currency symbols")
    if re.search(r"\b(urgent|claim|verify|suspended|click)\b", text, re.I):
        signals.append("urgency/action words")

    if not signals:
        signals_str = "No strong spam signals detected in this message."
    else:
        signals_str = "Signals found: " + ", ".join(signals) + "."

    level = "HIGH" if confidence > 0.95 else "MEDIUM" if confidence > 0.75 else "LOW"
    return f"Confidence level: {level}\n{signals_str}"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
EXAMPLES = [
    [
        "Congratulations! You've WON a FREE iPhone. CLICK HERE to claim your prize now: http://win.com"
    ],
    ["Hey, are we still on for dinner tonight? Let me know what time works for you."],
    ["URGENT: Your bank account has been SUSPENDED. Verify now at http://secure-login.biz"],
    ["Can you pick up some milk on your way home? Thanks!"],
    ["Free entry in 2 a wkly comp to win FA Cup Final tkts 21st May 2005. Text FA to 87121"],
    ["I'll be home by 7. Save me some food please :)"],
]

DESCRIPTION = """
## SMS Spam Classifier
Built with **DistilBERT** transfer learning — fine-tuned on 5,574 real SMS messages.

Type any SMS message below and the model will tell you if it's **spam** or **legitimate (ham)**.
"""


def build_interface() -> gr.Blocks:
    # theme and css moved to launch() in Gradio 6.0
    with gr.Blocks(title="SMS Spam Classifier") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=2):
                txt_input = gr.Textbox(
                    label="SMS Message",
                    placeholder="Type or paste an SMS message here…",
                    lines=4,
                    max_lines=8,
                )
                with gr.Row():
                    btn_classify = gr.Button("Classify", variant="primary", scale=2)
                    btn_clear = gr.Button("Clear", scale=1)

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=txt_input,
                    label="Try these examples",
                )

            with gr.Column(scale=1):
                out_label = gr.Textbox(label="Result", interactive=False, lines=1)
                out_bar = gr.Textbox(label="Confidence breakdown", interactive=False, lines=2)
                out_explain = gr.Textbox(label="Why?", interactive=False, lines=3)

        btn_classify.click(
            fn=classify,
            inputs=txt_input,
            outputs=[out_label, out_bar, out_explain],
        )
        btn_clear.click(
            fn=lambda: ("", "", "", ""),
            outputs=[txt_input, out_label, out_bar, out_explain],
        )
        txt_input.submit(
            fn=classify,
            inputs=txt_input,
            outputs=[out_label, out_bar, out_explain],
        )

        gr.Markdown(
            "_Model: DistilBERT fine-tuned on UCI SMS Spam Collection · "
            "Tracked with MLflow · Served via FastAPI + Gradio_"
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
