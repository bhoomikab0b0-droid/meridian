"""
News-sentiment feature powered by a pretrained transformer.

Uses ProsusAI/finbert (a BERT model fine-tuned specifically on financial
text) via Hugging Face `transformers` to score headlines as
positive/negative/neutral. This is the "pretrained best model" component
of the system: rather than training a sentiment classifier from scratch,
we reuse a model that has already learned financial language on a large
corpus, and fold its output in as an additional engineered feature
(`sentiment_score`) alongside the technical indicators.

Requires outbound network access to huggingface.co the first time it runs
(to download model weights, ~440MB, cached afterwards). Environments with
a restricted egress allowlist (no huggingface.co) will not be able to
download the weights; in that case this module transparently falls back
to a compact finance-aware lexicon scorer so the app keeps working, and
flags the fallback in its output so the UI can be honest about it.
"""
from __future__ import annotations

_pipeline = None
_load_attempted = False
_load_error = None

_POSITIVE_WORDS = {
    "beat", "beats", "surge", "surged", "rally", "rallied", "upgrade", "upgraded",
    "growth", "profit", "profits", "record", "strong", "outperform", "bullish",
    "gain", "gains", "soar", "soared", "expansion", "exceed", "exceeded", "buyback",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "missed", "plunge", "plunged", "downgrade", "downgraded",
    "loss", "losses", "weak", "underperform", "bearish", "slump", "slumped",
    "recall", "lawsuit", "fraud", "layoffs", "cut", "cuts", "warning", "default",
    "investigation", "bankrupt", "bankruptcy",
}


def _load_finbert():
    global _pipeline, _load_attempted, _load_error
    if _load_attempted:
        return _pipeline
    _load_attempted = True
    try:
        from transformers import pipeline
        _pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:  # network blocked, model not cached, transformers missing, etc.
        _load_error = str(e)
        _pipeline = None
    return _pipeline


def _lexicon_score(text: str) -> float:
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip(".,!?") in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?") in _NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def score_headlines(headlines: list[str]) -> dict:
    """
    Returns {"scores": [-1..1 per headline], "avg_score": float,
    "engine": "finbert-pretrained" | "lexicon-fallback"}
    """
    if not headlines:
        return {"scores": [], "avg_score": 0.0, "engine": "none"}

    model = _load_finbert()
    if model is not None:
        try:
            results = model(headlines, truncation=True)
            scores = []
            for r in results:
                sign = {"positive": 1, "negative": -1, "neutral": 0}.get(r["label"].lower(), 0)
                scores.append(round(sign * r["score"], 4))
            return {
                "scores": scores,
                "avg_score": round(sum(scores) / len(scores), 4),
                "engine": "finbert-pretrained",
            }
        except Exception:
            pass  # fall through to lexicon

    scores = [round(_lexicon_score(h), 4) for h in headlines]
    return {
        "scores": scores,
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "engine": "lexicon-fallback",
        "note": "FinBERT weights unavailable in this environment (no route to "
                "huggingface.co) -- using a lightweight finance lexicon instead. "
                "Deploy with normal internet access to enable the pretrained model.",
    }
