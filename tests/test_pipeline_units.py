import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import fuzzy_deduplicate, inject_indonesian_typos
from inference import ReviewPredictor


def test_fuzzy_deduplication():
    """Ensures MinHash LSH correctly identifies and drops near-duplicate texts."""
    df = pd.DataFrame({
        "text": [
            "pengiriman sangat cepat dan packing aman", 
            "pengiriman sangat cepat dan packing aman sekali ya",  # Near-duplicate
            "barang rusak parah tidak bisa dipakai sama sekali"    # Unique
        ]
    })
    
    # Using a slightly lower threshold for the short test strings
    deduped_df, dropped_count = fuzzy_deduplicate(df, threshold=0.5)
    
    assert len(deduped_df) == 2, "Failed to drop the near-duplicate review"
    assert dropped_count == 1, "Dropped count mismatch"
    assert "barang rusak" in deduped_df.iloc[1]["text"], "Kept the wrong unique review"


def test_typo_injection():
    """Ensures the robustness noise generator alters text without changing word counts."""
    original = "kualitas barang sangat bagus dan pengiriman cepat"
    
    # Force 100% noise level to guarantee mutation for the test
    noisy = inject_indonesian_typos(original, noise_level=1.0)
    
    assert original != noisy, "Typo injection failed to alter the string"
    assert len(original.split()) == len(noisy.split()), "Typo injection inappropriately added/removed whole words"


def test_inference_pipeline_artifacts_and_shapes():
    """Verifies that the trained artifacts load correctly and output expected tensor shapes."""
    models_dir = ROOT / "models"
    
    if not (models_dir / "baseline.joblib").exists():
        pytest.skip("Artifacts missing. Run `python src/train_baseline.py` before running this test.")
        
    predictor = ReviewPredictor(models_dir)
    
    # 1. Verify Labels
    assert len(predictor.labels) == 5, "Expected 5 emotion classes"
    assert "happy" in predictor.labels
    
    # 2. Verify Emotion Inference Shapes
    test_texts = ["pengiriman cepat sekali", "barang rusak kecewa"]
    probs = predictor.predict_proba(test_texts)
    
    assert type(probs) == np.ndarray
    assert probs.shape == (2, 5), "Output shape should be (n_samples, n_classes)"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be bounded between 0 and 1"
    
    # 3. Verify FastText Aspect Tagging
    aspects = predictor.predict_aspects(test_texts, threshold=0.1)
    
    assert len(aspects) == 2
    assert "primary" in aspects[0]
    assert "aspects" in aspects[0]
    assert type(aspects[0]["aspects"]) == dict