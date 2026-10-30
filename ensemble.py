"""
ensemble.py — V3+ backward-compatible shim.

Delegates to quant_core.DictModelAdapter for V3+ dict artifacts while
maintaining the original EnsembleModel interface for V2 .pkl files.
"""
import numpy as np
import joblib


class EnsembleModel:
    """
    Wraps LightGBM + Random Forest so the combined model exposes the
    standard sklearn interface: .predict_proba() and .classes_

    ONLY trades when BOTH models agree on the same class.
    When they disagree, returns CHOP (class 1) with CHOP confidence.

    In V3+, this class also serves as a backward-compatible wrapper.
    If initialized with a plain dict artifact, it delegates to
    quant_core.DictModelAdapter.
    """

    def __init__(self, pipeline_lgb=None, pipeline_rf=None, classes=None):
        self._is_v3_adapter = False
        self._adapter = None

        if isinstance(pipeline_lgb, dict):
            # V3+ plain-dict artifact passed directly
            from quant_core import DictModelAdapter
            self._adapter = DictModelAdapter(pipeline_lgb)
            self._is_v3_adapter = True
            self.classes_ = self._adapter.classes_
        else:
            self.pipeline_lgb = pipeline_lgb
            self.pipeline_rf = pipeline_rf
            self.classes_ = np.array(classes if classes is not None else [0, 1, 2])

    def predict_proba(self, X):
        if self._is_v3_adapter:
            return self._adapter.predict_proba(X)

        p_lgb = self.pipeline_lgb.predict_proba(X)
        p_rf = self.pipeline_rf.predict_proba(X)
        avg = (p_lgb + p_rf) / 2.0

        # Agreement check per sample
        lgb_class = p_lgb.argmax(axis=1)
        rf_class = p_rf.argmax(axis=1)
        disagree = (lgb_class != rf_class)

        # Force CHOP (class 1) when models disagree
        chop_idx = list(self.classes_).index(1) if 1 in self.classes_ else 1
        for i in np.where(disagree)[0]:
            avg[i] = 0.0
            avg[i, chop_idx] = 1.0

        return avg

    def predict(self, X):
        if self._is_v3_adapter:
            return self._adapter.predict(X)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        loaded = joblib.load(path)
        if isinstance(loaded, dict):
            return EnsembleModel(pipeline_lgb=loaded)
        return loaded
