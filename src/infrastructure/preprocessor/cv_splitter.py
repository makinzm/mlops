"""
CVSplitter — CV 分割戦略のインフラ実装。

UseCase 層が sklearn に直接依存しないよう、CV 分割ロジックを分離する。
"""

import numpy as np

from src.domain.data.table import DataFrame


class CVSplitter:
    """cv: 設定から sklearn を使って splits を生成する。"""

    def build(
        self,
        cv_cfg: dict[str, object],
        input_dfs: dict[str, DataFrame],
    ) -> list[tuple[list[int], list[int]]] | None:
        """cv_cfg と input_dfs から CV splits を生成して返す。strategy=none は None を返す。

        対応 strategy:
        - none: CV なし
        - kfold: sklearn KFold
        - time_series: sklearn TimeSeriesSplit
        - stratified_kfold: sklearn StratifiedKFold（target_col 必須）
        - group_kfold: sklearn GroupKFold（group_col 必須）
        - stratified_group_kfold: sklearn StratifiedGroupKFold（target_col + group_col 必須）
        - leave_one_group_out: sklearn LeaveOneGroupOut（group_col 必須）
        """
        strategy = str(cv_cfg.get("strategy", "none")) if cv_cfg else "none"
        if strategy == "none":
            return None
        if not input_dfs:
            return None

        input_id_raw = cv_cfg.get("input_id", None)
        if input_id_raw and str(input_id_raw) in input_dfs:
            target_df = input_dfs[str(input_id_raw)]
        else:
            target_df = next(iter(input_dfs.values()))

        n = len(target_df)
        n_splits = int(cv_cfg.get("n_splits", 5))  # ty:ignore[invalid-argument-type]

        if strategy == "kfold":
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return [
                (list(map(int, train_idx)), list(map(int, test_idx)))
                for train_idx, test_idx in kf.split(np.arange(n))
            ]

        if strategy == "time_series":
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=n_splits)
            return [
                (list(map(int, train_idx)), list(map(int, test_idx)))
                for train_idx, test_idx in tscv.split(np.arange(n))
            ]

        if strategy == "stratified_kfold":
            from sklearn.model_selection import StratifiedKFold

            target_col = str(cv_cfg.get("target_col", ""))
            y = target_df[target_col].to_list()
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            return [
                (list(map(int, train_idx)), list(map(int, test_idx)))
                for train_idx, test_idx in skf.split(range(n), y)
            ]

        if strategy == "group_kfold":
            from sklearn.model_selection import GroupKFold

            group_col = str(cv_cfg.get("group_col", ""))
            groups = target_df[group_col].to_list()
            gkf = GroupKFold(n_splits=n_splits)
            return [
                (list(map(int, train_idx)), list(map(int, test_idx)))
                for train_idx, test_idx in gkf.split(range(n), groups=groups)
            ]

        if strategy == "stratified_group_kfold":
            from sklearn.model_selection import StratifiedGroupKFold

            target_col = str(cv_cfg.get("target_col", ""))
            group_col = str(cv_cfg.get("group_col", ""))
            y = target_df[target_col].to_list()
            groups = target_df[group_col].to_list()
            sgkf = StratifiedGroupKFold(n_splits=n_splits)
            return [
                (list(map(int, train_idx)), list(map(int, test_idx)))
                for train_idx, test_idx in sgkf.split(range(n), y, groups=groups)
            ]

        if strategy == "leave_one_group_out":
            from sklearn.model_selection import LeaveOneGroupOut

            group_col = str(cv_cfg.get("group_col", ""))
            groups = target_df[group_col].to_list()
            logo = LeaveOneGroupOut()
            return [
                (list(map(int, train_idx)), list(map(int, test_idx)))
                for train_idx, test_idx in logo.split(range(n), groups=groups)
            ]

        return None
