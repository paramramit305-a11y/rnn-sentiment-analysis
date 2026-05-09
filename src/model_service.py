from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch

from src.config import settings
from src.preprocessing import clean_text
from src.rnn_model import SentimentRNN


class SentimentModelService:
    def __init__(self, model_path: Path | None = None, metadata_path: Path | None = None) -> None:
        self.model_path = Path(model_path or settings.model_path)
        self.vectorizer_path = Path(settings.vectorizer_path)
        self.label_map_path = Path(settings.label_map_path)
        self.metadata_path = Path(metadata_path or settings.metadata_path)
        self.model: SentimentRNN | None = None
        self.vectorizer: Any | None = None
        self.label_map: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}
        self.device = torch.device("cpu")

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. Run 'python train_rnn_model.py' first."
            )
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found at '{self.vectorizer_path}'. Run 'python train_rnn_model.py' first."
            )
        if not self.label_map_path.exists():
            raise FileNotFoundError(
                f"Label mapping file not found at '{self.label_map_path}'. Run 'python train_rnn_model.py' first."
            )

        self.vectorizer = joblib.load(self.vectorizer_path)
        self.label_map = json.loads(self.label_map_path.read_text(encoding="utf-8"))

        input_size = int(self.metadata.get("input_size", 0)) if self.metadata else None
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            input_size = int(self.metadata.get("input_size", 0))
        if not input_size:
            input_size = len(self.vectorizer.get_feature_names_out())

        hidden_size = int(self.metadata.get("hidden_size", 128))
        num_layers = int(self.metadata.get("num_layers", 1))

        self.model = SentimentRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {"model_name": "tfidf-rnn-pytorch"}

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "tfidf-rnn-pytorch"))

    def _ensure_loaded(self) -> None:
        if self.model is None or self.vectorizer is None:
            self.load()

    def predict(self, text: str) -> dict[str, Any]:
        self._ensure_loaded()
        cleaned_text = clean_text(text)

        if not cleaned_text:
            raise ValueError("Input text is empty after preprocessing.")

        vector = self.vectorizer.transform([cleaned_text]).toarray()
        tensor = torch.tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(1)
        with torch.no_grad():
            logits = self.model(tensor)
            positive_score = torch.sigmoid(logits).item()
        negative_score = 1.0 - positive_score

        score_map = {
            self.label_map.get("0", "negative"): float(negative_score),
            self.label_map.get("1", "positive"): float(positive_score),
        }
        sentiment = max(score_map, key=score_map.get)
        confidence = score_map[sentiment]

        response = {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "probabilities": {key: round(value, 4) for key, value in score_map.items()},
            "cleaned_text": cleaned_text,
            "model_name": self.model_name,
        }


        return response
