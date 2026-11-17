"""Prediction wrapper for the CPU-optimized LightGBM + FastText Pipeline.

Handles multi-label thresholding and FastText semantic aspect matching.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText
from huggingface_hub import snapshot_download
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

HF_REPO = "wiratheausom/NLP_aol"


def _fasttext_path() -> str:
    """Use the local Text model if present, otherwise fetch it from Hugging Face."""
    local = MODELS_DIR / "fasttext_aspects.model"
    if local.exists():
        return str(local)
    folder = snapshot_download(HF_REPO, allow_patterns=["fasttext_aspects.model*"])
    return str(Path(folder) / "fasttext_aspects.model")

class ReviewPredictor:
    display_name = "Calibrated LightGBM + FastText Pipeline"

    def __init__(self, models_dir: Path = MODELS_DIR):
        # Load ML Pipeline & Labels
        self.pipe = joblib.load(models_dir / "baseline.joblib")
        self.mlb = joblib.load(models_dir / "mlb.joblib")
        self.labels = list(self.mlb.classes_)
        
        # Load FastText & Meta
        self.ft_model = FastText.load(_fasttext_path(), mmap="r")
        self.meta = json.loads((models_dir / "baseline_meta.json").read_text(encoding="utf-8"))
        self.aspect_seeds = self.meta["aspect_seeds"]
        
        # Pre-compute average anchor vectors for each aspect
        self.aspect_vectors = {}
        for aspect, seeds in self.aspect_seeds.items():
            vecs = [self.ft_model.wv[w] for w in seeds if w in self.ft_model.wv]
            if vecs:
                avg_vec = np.mean(vecs, axis=0)
                self.aspect_vectors[aspect] = avg_vec / norm(avg_vec)

    def _get_text_vector(self, text: str) -> np.ndarray:
        words = text.lower().split()
        if not words:
            return np.zeros(self.ft_model.vector_size)
        # FastText handles OOV directly via subword n-grams
        vecs = [self.ft_model.wv[w] for w in words]
        avg_vec = np.mean(vecs, axis=0)
        norm_val = norm(avg_vec)
        return avg_vec / norm_val if norm_val > 0 else avg_vec

    def predict_aspects(self, texts: list[str], threshold: float = 0.4) -> list[dict]:
        """Tags aspects using semantic cosine similarity instead of rigid keywords."""
        results = []
        for text in texts:
            text_vec = self._get_text_vector(text)
            if norm(text_vec) == 0:
                results.append({"primary": "No aspect detected", "aspects": {}})
                continue
            
            sims = {}
            for aspect, anchor_vec in self.aspect_vectors.items():
                sim = np.dot(text_vec, anchor_vec)
                if sim >= threshold:
                    sims[aspect] = round(float(sim), 3)
            
            if not sims:
                results.append({"primary": "No aspect detected", "aspects": {}})
            else:
                # Sort by highest similarity
                sorted_sims = dict(sorted(sims.items(), key=lambda x: x[1], reverse=True))
                primary = list(sorted_sims.keys())[0]
                results.append({"primary": primary, "aspects": sorted_sims})
        return results

    def predict_proba(self, texts: list[str], progress: Callable[[float], None] | None = None) -> np.ndarray:
        probs = self.pipe.predict_proba(texts)
        if progress:
            progress(1.0)
        return probs


def load_predictor(models_dir: Path = MODELS_DIR) -> ReviewPredictor:
    return ReviewPredictor(models_dir)


def predict_dataframe(
    df: pd.DataFrame, 
    text_col: str, 
    predictor: ReviewPredictor, 
    progress: Callable[[float], None] | None = None
) -> pd.DataFrame:
    """Predict multi-label emotions + semantic aspects for a column of reviews."""
    texts = df[text_col].fillna("").astype(str).tolist()
    
    # 1. Predict Emotions (Multi-Label)
    probs = predictor.predict_proba(texts, progress=progress)
    labels = predictor.labels
    preds_bin = (probs >= 0.5).astype(int)
    
    predicted_emotions, confidences = [], []
    for i in range(len(texts)):
        active = [labels[j] for j, val in enumerate(preds_bin[i]) if val == 1]
        if not active:  # Fallback to max probability if nothing passes 0.5 threshold
            active = [labels[np.argmax(probs[i])]]
        predicted_emotions.append(", ".join(active))
        confidences.append(probs[i].max().round(4))
        
    out = pd.DataFrame({
        "review": texts,
        "predicted_emotions": predicted_emotions,
        "confidence": confidences
    })
    
    for j, lab in enumerate(labels):
        out[f"prob_{lab}"] = probs[:, j].round(4)
        
    # 2. Predict Semantic Aspects
    aspect_results = predictor.predict_aspects(texts)
    out["primary_aspect"] = [res["primary"] for res in aspect_results]
    out["all_aspects"] = [", ".join(res["aspects"].keys()) for res in aspect_results]
    out["aspect_similarities"] = [str(res["aspects"]) for res in aspect_results]
    
    return out