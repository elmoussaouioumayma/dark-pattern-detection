"""Text cleaning for dark pattern detection, shared by the notebook and app.py."""

import re

# Dark patterns often depend on numbers and symbols: "Only 2 left!", a countdown
# "0:44:56", "Save 20%", "$99.00". Instead of deleting them, they are replaced by
# special tokens so the model can still use them.
_TIMER = re.compile(r"\d{1,2}(?::\d{2}){1,3}")          # 0:44:56, 12:05
_PRICE = re.compile(r"[$€£]\s?\d[\d,.]*")                # $132.90
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")              # 20%, 12.5 %
_NUMBER = re.compile(r"\d[\d,.]*")                       # 2, 1,000


def clean_text(text):
    """Lowercase and keep numbers, prices, timers and '!' as special tokens."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)                   # HTML tags
    text = _TIMER.sub(" tok_timer ", text)
    text = _PRICE.sub(" tok_price ", text)
    text = _PERCENT.sub(" tok_percent ", text)
    text = _NUMBER.sub(" tok_num ", text)
    text = text.replace("!", " tok_excl ")
    text = re.sub(r"[^a-z_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text_simple(text):
    """The cleaning of the first version: letters only (numbers and symbols removed)."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
