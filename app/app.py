"""Tokopedia Review Emotion Dashboard - Streamlit app (Phase 6).

Run from the project root::
    streamlit run app/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference import load_predictor, predict_dataframe  # noqa: E402

DEMO_CSV = ROOT / "data" / "demo_reviews.csv"
EMOTIONS = ["anger", "fear", "happy", "love", "sadness"]
NEGATIVE_EMOTIONS = {"anger", "fear", "sadness"}
EMOTION_COLORS = {
    "anger": "#D62728", "fear": "#9467BD", "happy": "#E0A100",
    "love": "#E377C2", "sadness": "#4C78A8"
}
TEXT_COLUMN_GUESSES = ["customer review", "customer_review", "review", "ulasan", "text", "comment"]

EXAMPLES = [
    "Pengiriman super cepat, packing rapi pakai bubble wrap. Seller ramah banget!",
    "Barang rusak pas sampai dan admin tidak ada respon sama sekali. Kesal banget!",
    "Takut barangnya palsu karena harganya terlalu murah.",
    "Kualitasnya bagus banget, aku cinta sama produk ini, pasti beli lagi!",
    "Sedih, barang yang ditunggu ternyata cacat. Padahal buat kado.",
]

st.set_page_config(page_title="Tokopedia Review Dashboard", page_icon="💬", layout="wide")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading LightGBM + FastText Pipeline...")
def get_predictor():
    return load_predictor()


def probability_chart(labels: list[str], probs) -> "px.bar":
    frame = pd.DataFrame({"emotion": labels, "probability": probs})
    fig = px.bar(
        frame, x="probability", y="emotion", orientation="h", color="emotion",
        color_discrete_map=EMOTION_COLORS, text=frame["probability"].map("{:.1%}".format),
        category_orders={"emotion": EMOTIONS}
    )
    # Add a threshold line to show where multi-label cutoff occurs
    fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


# --------------------------------------------------------------------------- #
# Page 1 - Single Review Inspector
# --------------------------------------------------------------------------- #
def page_single() -> None:
    st.header("Single Review Inspector (Multi-Label)")
    st.caption("Enter a review to see all overlapping emotions and semantic aspect matching.")

    def _use_example() -> None:
        choice = st.session_state.get("example_choice")
        if choice and choice != "(write my own)":
            st.session_state["single_text"] = choice

    st.selectbox("Try an example", ["(write my own)"] + EXAMPLES, key="example_choice", on_change=_use_example)
    text = st.text_area("Review text", key="single_text", height=130)

    if not text.strip():
        st.info("Enter a review above to get a prediction.")
        return

    predictor = get_predictor()
    probs = predictor.predict_proba([text])[0]
    labels = predictor.labels
    
    # Thresholding for Multi-Label
    active_emotions = [l for i, l in enumerate(labels) if probs[i] >= 0.5]
    if not active_emotions:
        active_emotions = [labels[np.argmax(probs)]]
        
    aspects = predictor.predict_aspects([text])[0]

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.subheader("Detected Emotion(s)")
        for em in active_emotions:
            st.markdown(
                f"<div style='font-size:1.8rem;font-weight:700;color:{EMOTION_COLORS.get(em, '#333')}'>{em.upper()}</div>",
                unsafe_allow_html=True,
            )
        st.metric("Max Confidence", f"{probs.max():.1%}")

        st.subheader("Primary Aspect (Semantic Match)")
        if aspects["primary"] == "No aspect detected":
            st.warning("No aspects reached the FastText similarity threshold.")
        else:
            st.success(aspects["primary"])
            with st.expander("Vector Similarities"):
                for aspect, score in aspects["aspects"].items():
                    st.write(f"**{aspect}**: {score:.3f} cosine sim")
    with right:
        st.subheader("Probability per emotion")
        st.plotly_chart(probability_chart(labels, probs), width="stretch", key="single_prob_chart")


# --------------------------------------------------------------------------- #
# Page 2 - Batch Seller Dashboard
# --------------------------------------------------------------------------- #
def render_batch_results(res: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Batch Results")
    
    n = len(res)
    # Explode multi-labels so a review tagged "happy, love" counts for both
    exploded = res.assign(emotion=res['predicted_emotions'].str.split(', ')).explode('emotion')
    counts = exploded["emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
    
    negative_count = exploded[exploded['emotion'].isin(NEGATIVE_EMOTIONS)]['review'].nunique()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analysed", f"{n:,}")
    c2.metric("Most common emotion", f"{counts.idxmax()}")
    c3.metric("Negative sentiment (Reviews)", f"{negative_count / n:.0%}", help="Has anger, fear, or sadness")
    c4.metric("Average max confidence", f"{res['confidence'].mean():.0%}")

    st.markdown("#### Overall Emotion Breakdown (Multi-Label)")
    frame = counts.rename_axis("emotion").reset_index(name="tags")
    frame["share_of_reviews"] = frame["tags"] / n
    fig = px.bar(
        frame, x="emotion", y="tags", color="emotion", color_discrete_map=EMOTION_COLORS,
        text=frame.apply(lambda r: f"{int(r['tags'])} ({r['share_of_reviews']:.0%})", axis=1),
        category_orders={"emotion": EMOTIONS},
    )
    fig.update_layout(showlegend=False, height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Emotion Tags")
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch", key="batch_emotion_chart")

    st.markdown("#### Aspect x Emotion Matrix")
    matrix = pd.crosstab(exploded['primary_aspect'], exploded['emotion'])
    heat = px.imshow(
        matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues",
        labels=dict(x="Emotion", y="Primary Aspect", color="Tags")
    )
    heat.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(heat, width="stretch", key="batch_matrix_chart")

    st.markdown("#### Predictions Table")
    st.dataframe(
        res, width="stretch", hide_index=True,
        column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.2f")},
    )


def page_batch() -> None:
    st.header("Batch Seller Dashboard")
    st.caption("Upload a CSV of reviews to run FastText aspect matching and Multi-Label emotion prediction.")

    uploaded = st.file_uploader("CSV file with a review text column", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state["batch_input"] = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            st.session_state["batch_input_name"] = uploaded.name
        except Exception as exc: 
            st.error(f"Could not read this file: {exc}")

    if DEMO_CSV.exists() and st.button("Load the demo reviews"):
        st.session_state["batch_input"] = pd.read_csv(DEMO_CSV)
        st.session_state["batch_input_name"] = DEMO_CSV.name

    data = st.session_state.get("batch_input")
    if data is None:
        return

    st.write(f"**{st.session_state.get('batch_input_name', 'data')}** - {len(data):,} rows")
    
    # Guess text column
    lower_cols = [str(c).lower() for c in data.columns]
    idx = 0
    for guess in TEXT_COLUMN_GUESSES:
        if guess in lower_cols:
            idx = lower_cols.index(guess)
            break
            
    column = st.selectbox("Which column contains the review text?", list(data.columns), index=idx)
    st.dataframe(data.head(3), width="stretch", hide_index=True)

    if st.button("Run analysis", type="primary"):
        predictor = get_predictor()
        bar = st.progress(0.0, text="Predicting Multi-Label & Aspects...")
        results = predict_dataframe(data, column, predictor, progress=lambda p: bar.progress(min(float(p), 1.0)))
        bar.empty()
        st.session_state["batch_results"] = results

    if "batch_results" in st.session_state:
        render_batch_results(st.session_state["batch_results"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("Tokopedia CPU-Optimized Dashboard")
    
    if not (ROOT / "models" / "baseline.joblib").exists():
        st.error("Model not found. Run `python src/data_loader.py` then `python src/train_baseline.py` first.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        st.success("Active Engine:\nSubword LightGBM + FastText")
        page = st.radio("Page", ["Single Review Inspector", "Batch Seller Dashboard"], key="page")

    if page == "Single Review Inspector":
        page_single()
    else:
        page_batch()

if __name__ == "__main__":
    main()