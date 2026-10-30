"""
train_model.py — V3+ deprecation guard.

Local 780-bar training is disabled in V3+ to protect the 10-year cloud
model artifact. Run colab_cloud_trainer.py in Google Colab instead.

This module preserves backward compatibility:
  - METADATA_PATH is still importable by app.py
  - Running this file directly raises a protective RuntimeError
"""
import json
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / 'stock_price_model.pkl'
MODEL_PATH_V2 = Path(__file__).resolve().parent / 'stock_price_model_v2.pkl'
METADATA_PATH = Path(__file__).resolve().parent / 'model_metadata.json'


def train_stock_model(ticker='AAPL', start_date='2020-01-01', end_date=None):
    """Refuse to train locally if a V3+ model artifact exists.

    This guard prevents the 72-hour ensure_fresh_model() auto-retrain
    from overwriting the 10-year cloud-trained model with a 780-bar
    Yahoo Finance model.
    """
    if MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            if isinstance(artifact, dict) and artifact.get('model_version') == 'V3+':
                raise RuntimeError(
                    "Local 780-bar training is disabled in V3+ to protect "
                    "your 10-year cloud model. Run colab_cloud_trainer.py "
                    "in Google Colab instead."
                )
        except (RuntimeError,):
            raise
        except Exception:
            pass  # Not a V3+ artifact, allow legacy training

    raise RuntimeError(
        "Local training has been deprecated in V3+. "
        "Upload colab_cloud_trainer.py to Google Colab and train there."
    )


if __name__ == '__main__':
    train_stock_model()
