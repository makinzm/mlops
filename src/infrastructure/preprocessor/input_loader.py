"""
InputLoader — inputs 設定から DataFrame を読み込むインフラ実装。

UseCase 層が polars に直接依存しないよう、ファイル読み込みロジックを分離する。
"""

from pathlib import Path

import polars as pl

from src.domain.data.table import DataFrame


class InputLoader:
    """inputs 設定リストから polars DataFrame を読み込む。"""

    def load(self, inputs_raw: list[object]) -> dict[str, DataFrame]:
        """inputs 設定リストから DataFrame を読み込んで返す。

        Args:
            inputs_raw: OmegaConf.to_container() で変換済みの inputs リスト。
                        各要素は {"id": str, "path": str, "format": str} の dict。
        """
        input_dfs: dict[str, DataFrame] = {}
        for inp in inputs_raw:
            inp_dict = dict(inp)  # ty:ignore[no-matching-overload]
            inp_id = str(inp_dict["id"])
            fmt = str(inp_dict.get("format", "csv"))
            if "path" in inp_dict:
                path = Path(str(inp_dict["path"]))
                if fmt == "parquet":
                    input_dfs[inp_id] = pl.read_parquet(path)
                else:
                    input_dfs[inp_id] = pl.read_csv(path)
        return input_dfs
