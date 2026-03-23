"""
LightGBM Trainer — Trainer Protocol の実装。

設計:
  fit_folds() が fold ごとに以下を実行する:
    1. preprocess_output_dir/fold_{N}/train.parquet / test.parquet を読み込み
    2. サンプル重みを構築（sample_weight_col > class_weight > is_unbalance）
    3. lgb.Dataset を作成
    4. lgb.train() — early stopping
    5. fold_{N}/model.lgbm に保存
    6. validation セットで予測 → oof_train.parquet
    7. error_analysis.parquet（TP/TN/FP/FN サンプリング）
    8. feature_importance.parquet
  CV スコアを集計して TrainResult を返す。

  timestamp / commit_hash は TrainUseCase が生成して cfg に含めて渡す。
  (_timestamp, _commit_hash キー)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig

from src.domain.model.trainer import FoldResult, TrainResult


class LightGBMTrainer:
    """LightGBM を使った k-fold クロスバリデーション学習器。"""

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def fit_folds(
        self,
        preprocess_output_dir: Path,
        output_dir: Path,
        cfg: dict[str, Any],
    ) -> TrainResult:
        """fold ごとに学習し TrainResult を返す。

        output_dir は TrainUseCase が生成した {job_id}/{timestamp}/ を受け取る。
        fold_{N}/ ディレクトリはこの直下に作成される。
        """
        dcfg = self._cfg
        fold_dirs = sorted(preprocess_output_dir.glob("fold_*"))
        if not fold_dirs:
            raise ValueError(f"fold ディレクトリが見つかりません: {preprocess_output_dir}")

        # timestamp / commit_hash は UseCase 側が生成して cfg に含めて渡す
        timestamp: str = cfg.get("_timestamp", datetime.now().strftime("%Y%m%dT%H%M%S"))
        commit_hash: str = cfg.get("_commit_hash", "unknown")

        job_id: str = cfg.get("job_id", "lgbm")
        target_col: str = dcfg.target_col
        feature_cols: list[str] = list(dcfg.feature_cols)
        cat_features: list[str] = list(dcfg.get("categorical_feature", []))
        weight_col: str | None = dcfg.get("sample_weight_col")
        metric: str = dcfg.loss.metric
        seed: int = int(dcfg.seed)
        n_error: int = int(dcfg.report.n_error_samples)

        fold_results: list[FoldResult] = []

        for fold_dir in fold_dirs:
            fold_idx = int(fold_dir.name.replace("fold_", ""))
            fold_out = output_dir / f"fold_{fold_idx}"
            fold_out.mkdir(parents=True, exist_ok=True)

            train_df = pl.read_parquet(fold_dir / "train.parquet").to_pandas()
            valid_df = pl.read_parquet(fold_dir / "test.parquet").to_pandas()

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_valid = valid_df[feature_cols]
            y_valid = valid_df[target_col]

            # サンプル重み
            train_weights = _build_weights(train_df, y_train, weight_col, dcfg)

            # categorical feature のエンコード（lgbm は category dtype を使う）
            # feature_cols に含まれるものだけを対象にする
            active_cat_features = [c for c in cat_features if c in X_train.columns]
            if active_cat_features:
                X_train = X_train.copy()
                X_valid = X_valid.copy()
                for col in active_cat_features:
                    X_train[col] = X_train[col].astype("category")
                    X_valid[col] = X_valid[col].astype("category")

            dtrain = lgb.Dataset(
                X_train,
                label=y_train,
                weight=train_weights,
                categorical_feature=active_cat_features if active_cat_features else "auto",
            )
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

            params = _build_lgbm_params(dcfg, seed)
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=int(dcfg.lgbm.get("early_stopping_rounds", 50)),
                    verbose=False,
                ),
                lgb.log_evaluation(period=int(dcfg.logging.get("eval_freq", 100))),
            ]

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=int(dcfg.lgbm.n_estimators),
                valid_sets=[dtrain, dvalid],
                valid_names=["train", "valid"],
                callbacks=callbacks,  # ty:ignore[invalid-argument-type]
            )

            # スコア取得
            train_score = float(booster.best_score.get("train", {}).get(metric, 0.0))
            valid_score = float(booster.best_score.get("valid", {}).get(metric, 0.0))
            best_iter = int(booster.best_iteration)

            # model 保存（LightGBM のテキスト形式 → .lgbm）
            model_path = fold_out / "model.lgbm"
            booster.save_model(str(model_path))

            # OOF 予測保存
            valid_preds: np.ndarray = np.asarray(booster.predict(X_valid, num_iteration=best_iter))
            oof_df = valid_df[[target_col] + feature_cols].copy()
            oof_df["predicted_proba"] = valid_preds
            oof_path = fold_out / "oof_train.parquet"
            pl.from_pandas(oof_df).write_parquet(oof_path)

            # error_analysis 生成
            ea_path = fold_out / "error_analysis.parquet"
            _write_error_analysis(
                valid_df, valid_preds, y_valid, target_col, feature_cols, n_error, ea_path
            )

            # feature importance 保存
            fi_path: Path | None = None
            fi: dict[str, float] = {}
            if dcfg.logging.get("save_importance", True):
                fi_path = fold_out / "feature_importance.parquet"
                fi = dict(
                    zip(
                        booster.feature_name(),
                        booster.feature_importance(importance_type="gain").tolist(),
                    )
                )
                pl.DataFrame(
                    {"feature": list(fi.keys()), "importance": list(fi.values())}
                ).write_parquet(fi_path)

            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    train_score=train_score,
                    valid_score=valid_score,
                    metric=metric,
                    model_path=model_path,
                    oof_path=oof_path,
                    error_analysis_path=ea_path,
                    feature_importance_path=fi_path,
                    n_train=len(X_train),
                    n_valid=len(X_valid),
                    best_iteration=best_iter,
                    feature_importances=fi,
                )
            )

        scores = [f.valid_score for f in fold_results]
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))

        return TrainResult(
            job_id=job_id,
            timestamp=timestamp,
            commit_hash=commit_hash,
            trainer_type="lgbm",
            output_dir=output_dir,
            fold_results=fold_results,
            cv_mean_score=cv_mean,
            cv_std_score=cv_std,
            metric=metric,
            seed=seed,
        )


# ──────────────────────────────────────────────────────────────
# ヘルパー関数
# ──────────────────────────────────────────────────────────────


def _build_weights(
    df: pd.DataFrame,
    y: pd.Series,
    weight_col: str | None,
    cfg: DictConfig,
) -> np.ndarray | None:
    """サンプル重みを構築する。優先順位: sample_weight_col > class_weight > is_unbalance。

    is_unbalance は lgb.train() の params で処理するため、ここでは None を返す。
    """
    if weight_col and weight_col in df.columns:
        return df[weight_col].to_numpy()

    class_weight = cfg.loss.get("class_weight")
    if class_weight is not None:
        weights = np.ones(len(y), dtype=float)
        for cls_val, w in class_weight.items():
            weights[y == int(cls_val)] = float(w)
        return weights

    return None


def _build_lgbm_params(cfg: DictConfig, seed: int) -> dict[str, Any]:
    """LightGBM params dict を構築する。"""
    lgbm_cfg = cfg.lgbm
    loss_cfg = cfg.loss
    n_jobs = int(cfg.environment.get("n_jobs", -1))

    params: dict[str, Any] = {
        "objective": loss_cfg.objective,
        "metric": loss_cfg.metric,
        "seed": seed,
        "num_threads": n_jobs,
        "num_leaves": int(lgbm_cfg.get("num_leaves", 31)),
        "max_depth": int(lgbm_cfg.get("max_depth", -1)),
        "learning_rate": float(lgbm_cfg.get("learning_rate", 0.05)),
        "feature_fraction": float(lgbm_cfg.get("feature_fraction", 0.9)),
        "bagging_fraction": float(lgbm_cfg.get("bagging_fraction", 0.8)),
        "bagging_freq": int(lgbm_cfg.get("bagging_freq", 5)),
        "min_child_samples": int(lgbm_cfg.get("min_child_samples", 20)),
        "reg_alpha": float(lgbm_cfg.get("reg_alpha", 0.0)),
        "reg_lambda": float(lgbm_cfg.get("reg_lambda", 0.0)),
        "verbose": int(lgbm_cfg.get("verbose", -1)),
    }

    if loss_cfg.get("is_unbalance", False):
        params["is_unbalance"] = True

    return params


def _write_error_analysis(
    df: pd.DataFrame,
    preds: np.ndarray,
    y: pd.Series,
    target_col: str,
    feature_cols: list[str],
    n_samples: int,
    out_path: Path,
) -> None:
    """4 種サンプリング（TP/TN/FP/FN）を out_path に保存する。"""
    threshold = 0.5
    pred_label = (preds >= threshold).astype(int)

    result = df[feature_cols].copy()
    result["target"] = y.values
    result["predicted_proba"] = preds
    result["predicted_label"] = pred_label
    result["is_correct"] = (pred_label == y.values).astype(int)
    result["error_magnitude"] = np.abs(preds - y.values)

    def _label(t: int, p: int) -> str:
        if t == 1 and p == 1:
            return "TP"
        if t == 0 and p == 0:
            return "TN"
        if t == 0 and p == 1:
            return "FP"
        return "FN"

    result["sample_type"] = [_label(int(t), int(p)) for t, p in zip(y.values, pred_label)]

    samples: list[pd.DataFrame] = []
    for stype in ("TP", "TN", "FP", "FN"):
        subset = result[result["sample_type"] == stype]
        if stype in ("FP", "FN"):
            subset = subset.sort_values("error_magnitude", ascending=False)
        else:
            subset = subset.sort_values("error_magnitude", ascending=True)
        samples.append(subset.head(n_samples))

    combined = pd.concat(samples, ignore_index=True)
    pl.from_pandas(combined).write_parquet(out_path)
