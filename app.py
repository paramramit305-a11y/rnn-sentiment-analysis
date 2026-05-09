from __future__ import annotations


import streamlit as st

from src.model_service import SentimentModelService


st.set_page_config(page_title="Sentiment Analysis App", page_icon="🎬", layout="centered")
st.title("IMDb Sentiment Analysis")
st.caption("Single-file Streamlit deployment using your trained NLP + RNN sentiment model.")


@st.cache_resource
def get_model_service() -> SentimentModelService:
    service = SentimentModelService()
    service.load()
    return service


with st.sidebar:
    st.subheader("Settings")
    st.markdown("This app predicts whether a review is positive or negative using a PyTorch RNN model.")


review_text = st.text_area(
    "Enter a review",
    height=220,
    placeholder="Type a movie review or any opinion text here...",
)


if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter some review text first.")
    else:
        try:
            with st.spinner("Analyzing sentiment..."):
                service = get_model_service()
                result = service.predict(text=review_text)

            sentiment = result["sentiment"]
            confidence = float(result["confidence"])
            probabilities = result["probabilities"]

            if sentiment == "positive":
                st.success(f"Prediction: Positive ({confidence:.2%} confidence)")
            else:
                st.error(f"Prediction: Negative ({confidence:.2%} confidence)")

            st.subheader("Probability Scores")
            st.json(probabilities)

            with st.expander("Cleaned Text"):
                st.write(result["cleaned_text"])


            st.caption(f"Model: {result.get('model_name', 'unknown')}")
        except FileNotFoundError as exc:
            st.error(f"Model file missing: {exc}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
