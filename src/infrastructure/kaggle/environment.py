"""
KaggleEnvironment — Kaggle Notebook 環境の検出とパス解決。

Kaggle Notebook では input/output のルートパスが固定されており、
ローカル環境とパスの規則が異なる。このモジュールはその差異を吸収する薄いアダプター層。

パス対応:
  input:  ローカル Path(slug)           → Kaggle /kaggle/input/competitions/{slug}
  output: ローカル Path(".")            → Kaggle /kaggle/working

環境判定:
  KAGGLE_KERNEL_RUN_TYPE 環境変数が設定されている場合を Kaggle 環境とみなす。
  値は "Interactive"（手動実行）または "Batch"（スケジュール実行）。
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_KEY = "KAGGLE_KERNEL_RUN_TYPE"
_KAGGLE_COMPETITION_ROOT = Path("/kaggle/input/competitions")
_KAGGLE_OUTPUT_ROOT = Path("/kaggle/working")


class KaggleEnvironment:
    """Kaggle Notebook 環境の検出とパス解決を行うアダプター層。

    Hydra Config を直接書き換えず、実行時にパスを解決する薄いラッパー。
    全メソッドを staticmethod にすることで、インスタンス化不要で利用できる。
    """

    @staticmethod
    def is_kaggle_notebook() -> bool:
        """現在の実行環境が Kaggle Notebook かどうかを返す。

        KAGGLE_KERNEL_RUN_TYPE 環境変数が設定されている場合に True を返す。
        ローカル環境では通常この環境変数は設定されない。

        Returns:
            Kaggle Notebook 環境なら True、ローカル環境なら False。
        """
        return os.environ.get(_ENV_KEY) is not None

    @staticmethod
    def resolve_input_root(dataset_slug: str) -> Path:
        """input データのルートパスを返す。

        Args:
            dataset_slug: competition/dataset の slug 名。
                          例: "my-competition", "house-prices-advanced-regression-techniques"

        Returns:
            Kaggle 環境なら /kaggle/input/competitions/{slug}、
            ローカル環境なら Path(dataset_slug)。
        """
        if KaggleEnvironment.is_kaggle_notebook():
            return _KAGGLE_COMPETITION_ROOT / dataset_slug
        return Path(dataset_slug)

    @staticmethod
    def resolve_output_root() -> Path:
        """output データのルートパスを返す。

        Returns:
            Kaggle 環境なら /kaggle/working、
            ローカル環境なら Path(".")。
        """
        if KaggleEnvironment.is_kaggle_notebook():
            return _KAGGLE_OUTPUT_ROOT
        return Path(".")
