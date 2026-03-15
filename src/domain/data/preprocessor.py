"""
前処理パイプラインのコアデータ構造。

全レイヤー（Resolver / DAGRunner / UseCase）が依存するドメインモデル。
インフラ層への依存を持たない純粋なデータクラスのみを定義する。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepResult:
    """1ステップの実行結果。

    status は "ok" / "skipped" / "failed" の3値。
    - "ok"      : 正常終了
    - "skipped" : Resolver 未登録 / Method 未実装のため省略
    - "failed"  : 実行中に例外が発生したが、後続ステップは継続
    """

    resolver: str
    method: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str | None = None


@dataclass
class ColumnMeta:
    """出力カラムのモダリティ情報。

    modality は "tabular" / "image_embed" / "text_embed" / "sequence" など。
    dtype は "float32" / "int64" / "List[float32]" など Polars 互換の文字列表現。
    """

    name: str
    modality: str
    dtype: str


@dataclass
class Node:
    """DAG の1ノード。

    - Input Node (is_input=True): inputs: で定義されるデータソース
    - Transform Node: steps: の各要素
    - Output Node: resolver_cfg に "output" キーを持つ Transform Node

    from_nodes: このノードが依存するノード id のリスト。
      - 空リスト → 直前ノードを自動使用（DAGRunner が解決）
      - 単一要素 → 線形フロー
      - 複数要素 → マージ（join/concat）
    """

    id: str
    resolver_cfg: dict[str, Any]
    from_nodes: list[str] = field(default_factory=list)
    is_input: bool = False


@dataclass
class PreprocessResult:
    """前処理パイプライン全体の実行結果。

    UseCase が返し、学習コードや後続ジョブが参照するマニフェスト情報。
    preprocess_result.yaml に対応する。
    """

    output_path: Path
    columns: list[ColumnMeta]
    n_rows: int | None
    n_splits: int | None
    step_results: list[StepResult]
    commit_hash: str
    seed: int
    executor_used: str = "local"
    executor_fallback: bool = False
    executor_requested: str | None = None
