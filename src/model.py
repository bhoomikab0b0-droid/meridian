"""
Core prediction engine.

Architecture (deliberately upgraded from a plain RandomForest+Stacking setup):

  Layer 0 (base learners, direction of next `horizon`-bar move):
    - XGBoost classifier      (gradient boosted trees, handles nonlinearity well)
    - LightGBM classifier     (fast, leaf-wise boosting, good on tabular financial data)
    - RandomForest classifier (variance reduction, robust to noisy features)
    - ExtraTrees classifier   (extra randomization, decorrelates from RF)

  Layer 1 (meta learner):
    - Logistic Regression trained on out-of-fold Layer-0 probabilities
      (proper stacking via walk-forward OOF predictions -- no leakage)

  Layer 2 (meta-labeling / confidence gate):
    - A second LightGBM model trained to predict whether the Layer-1 call
      should be *acted on* (Lopez de Prado meta-labeling). Its output is
      used both to size positions and to filter low-conviction trades out
      of the backtest, which is the main lever that improves Sharpe without
      needing a higher raw directional accuracy.

  Uncertainty:
    - Because the ensemble already contains 4 diverse base learners, we
      also report their prediction agreement (fraction that agree on
      direction) as a model-uncertainty score shown in the UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb


class StackedDirectionalModel:
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        n_jobs controls internal thread parallelism for every base learner.
        When the walk-forward backtest fits many windows *concurrently* in
        separate processes (see src/backtest.py), each of those processes
        should pass n_jobs=1 here -- otherwise every process would also try
        to spawn as many threads as there are CPU cores, oversubscribing the
        machine (N windows x N cores each). A single ad-hoc fit (e.g. the
        "live" model trained once on full history for today's signal) can
        safely leave this at -1 to use all cores.
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scaler = StandardScaler()
        self.base_models = {
            "xgb": xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                min_child_weight=5, random_state=random_state,
                eval_metric="logloss", n_jobs=n_jobs,
            ),
            "lgbm": lgb.LGBMClassifier(
                n_estimators=350, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                min_child_samples=25, random_state=random_state,
                n_jobs=n_jobs, verbose=-1,
            ),
            "rf": RandomForestClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=20,
                random_state=random_state, n_jobs=n_jobs,
            ),
            "et": ExtraTreesClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=20,
                random_state=random_state, n_jobs=n_jobs,
            ),
        }
        self.meta_model = LogisticRegression(max_iter=1000, C=0.5)
        self.confidence_gate = lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=random_state, n_jobs=n_jobs, verbose=-1,
        )
        self.feature_cols: list[str] = []
        self.fitted = False

    # ------------------------------------------------------------------
    def _oof_predictions(self, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> np.ndarray:
        """Walk-forward out-of-fold Layer-0 probabilities (no leakage into meta-learner)."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof = np.zeros((len(X), len(self.base_models)))
        for name_idx, (name, model) in enumerate(self.base_models.items()):
            col = np.full(len(X), np.nan)
            for train_idx, val_idx in tscv.split(X):
                m = self._clone(model)
                m.fit(X[train_idx], y[train_idx])
                col[val_idx] = m.predict_proba(X[val_idx])[:, 1]
            oof[:, name_idx] = col
        return oof

    @staticmethod
    def _clone(model):
        from sklearn.base import clone
        return clone(model)

    # ------------------------------------------------------------------
    def fit(self, df_features: pd.DataFrame, y: pd.Series):
        """
        y: raw signed label series (e.g. triple-barrier label: +1/-1/0).
        Meta-labels for the confidence gate are derived internally from
        each row's true out-of-fold prediction correctness -- no external
        meta_y needed, which avoids any chance of leaking future
        information into the label the gate is trained on.
        """
        self.feature_cols = list(df_features.columns)
        mask = df_features.notna().all(axis=1) & y.notna()
        X = df_features.loc[mask].values
        y_signed = y.loc[mask].values
        y_arr = (y_signed > 0).astype(int)  # direction: up=1, down/flat=0

        X_scaled = self.scaler.fit_transform(X)

        # Layer 0 -> Layer 1 stacking, leak-free via OOF
        oof = self._oof_predictions(X_scaled, y_arr)
        valid_rows = ~np.isnan(oof).any(axis=1)
        self.meta_model.fit(oof[valid_rows], y_arr[valid_rows])

        # Refit base models on full data for future inference
        for name, model in self.base_models.items():
            model.fit(X_scaled, y_arr)

        # Layer 2 (meta-labeling): was the OOF stacked prediction actually
        # correct on this historical row? That becomes the label the
        # confidence gate learns to anticipate.
        meta_probs = self.meta_model.predict_proba(oof[valid_rows])[:, 1]
        oof_direction = np.where(meta_probs >= 0.5, 1, 0)
        gate_labels = (oof_direction == y_arr[valid_rows]).astype(int)

        gate_features = np.column_stack([
            meta_probs,
            np.abs(meta_probs - 0.5) * 2,          # distance from indecision
            oof[valid_rows].std(axis=1),            # base-learner disagreement
        ])
        # Guard against a degenerate single-class target (can happen on
        # short/unlucky windows) -- confidence gate falls back to "always
        # trust the signal" rather than raising.
        if len(np.unique(gate_labels)) < 2:
            self.confidence_gate = None
        else:
            self.confidence_gate.fit(gate_features, gate_labels)

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, df_features: pd.DataFrame) -> dict:
        """Predict on the most recent row(s). Returns dict of arrays."""
        X = df_features[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        base_probs = np.column_stack([
            model.predict_proba(X_scaled)[:, 1] for model in self.base_models.values()
        ])
        meta_prob = self.meta_model.predict_proba(base_probs)[:, 1]
        agreement = 1 - base_probs.std(axis=1) * 2  # crude agreement score in [~0,1]
        agreement = np.clip(agreement, 0, 1)

        if self.confidence_gate is None:
            confidence = np.full(len(X), 0.6)  # neutral fallback confidence
        else:
            gate_features = np.column_stack([
                meta_prob,
                np.abs(meta_prob - 0.5) * 2,
                base_probs.std(axis=1),
            ])
            confidence = self.confidence_gate.predict_proba(gate_features)[:, 1]

        direction = np.where(meta_prob >= 0.5, 1, -1)
        return {
            "direction": direction,
            "up_probability": meta_prob,
            "base_learner_probs": base_probs,
            "agreement": agreement,
            "trade_confidence": confidence,  # meta-labeling: should this signal be acted on
        }

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        imps = []
        for name, model in self.base_models.items():
            if hasattr(model, "feature_importances_"):
                imps.append(pd.Series(model.feature_importances_, index=self.feature_cols, name=name))
        if not imps:
            return pd.DataFrame()
        combined = pd.concat(imps, axis=1)
        combined["avg"] = combined.mean(axis=1)
        return combined.sort_values("avg", ascending=False).head(top_n)
