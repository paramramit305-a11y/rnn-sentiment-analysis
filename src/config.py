from __future__ import annotations

from dotenv import load_dotenv
import os

load_dotenv()

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    app_name: str = "IMDb Sentiment Analysis"
    model_path: Path = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "rnn_sentiment_model.pt"))
    vectorizer_path: Path = Path(os.getenv("VECTORIZER_PATH", ROOT_DIR / "models" / "tfidf_vectorizer.joblib"))
    label_map_path: Path = Path(os.getenv("LABEL_MAP_PATH", ROOT_DIR / "models" / "label_mapping.json"))
    metadata_path: Path = Path(os.getenv("MODEL_METADATA_PATH", ROOT_DIR / "models" / "model_metadata.json"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


settings = Settings()
