"""
DAG 実行エンジン。

責務:
1. Node リストから依存グラフを構築する
2. targets から逆向きに必要な Node だけを特定する（部分実行）
3. トポロジカル順にノードを実行する
4. from: の自動解決（省略時は直前ノードを使う）
5. step_results の収集

フロー:
    input_dfs → [Input Nodes] → [Transform Nodes] → results
"""

from pathlib import Path

import polars as pl

from src.domain.data.preprocessor import Node, StepResult
from src.infrastructure.preprocessor.registry import run_step


class DAGRunner:
    """DAG の依存解決と順次実行を担うエンジン。"""

    def __init__(
        self,
        nodes: list[Node],
        input_dfs: dict[str, pl.DataFrame],
        output_dir: Path,
        cv_splits: list[tuple[list[int], list[int]]] | None,
    ) -> None:
        self._nodes = nodes
        self._input_dfs = input_dfs
        self._output_dir = output_dir
        self._cv_splits = cv_splits
        self._step_results: list[StepResult] = []
        # id → Node のインデックス
        self._node_map: dict[str, Node] = {n.id: n for n in nodes}

    def run(self, targets: list[str]) -> dict[str, pl.DataFrame]:
        """targets を末尾として必要な Node だけを実行し、結果を返す。

        Returns:
            {node_id: DataFrame} — 実行されたノードの結果マップ
        """
        # 必要なノードを特定（targets から逆向きに依存解決）
        required = self._resolve_required(targets)

        # from: の自動解決（省略時は直前ノードを使う）
        self._resolve_auto_from(required)

        # トポロジカルソート
        execution_order = self._topological_sort(required)

        # Input Nodes を初期化
        cache: dict[str, pl.DataFrame] = dict(self._input_dfs)

        # 変換ノードを順次実行
        for node_id in execution_order:
            node = self._node_map[node_id]
            if node.is_input:
                continue  # Input Node は既に cache に入っている

            # 入力 DataFrame の取得
            if len(node.from_nodes) == 0:
                # from_nodes が空 = 入力なし（通常は自動解決済みのはず）
                raise RuntimeError(f"Node '{node_id}' has no input resolved")
            elif len(node.from_nodes) == 1:
                input_df = cache[node.from_nodes[0]]
            else:
                # 複数入力 → 最初の DF を base に後続を join（resolver に任せる）
                input_df = cache[node.from_nodes[0]]

            # Resolver 名と kwargs を node の resolver_cfg から取り出す
            resolver_name, method, kwargs = self._parse_resolver_cfg(node, cache)

            # ステップ実行
            result_df, step_result = run_step(
                df=input_df,
                resolver_name=resolver_name,
                method=method,
                kwargs=kwargs,
            )
            self._step_results.append(step_result)
            cache[node_id] = result_df

        return {node_id: cache[node_id] for node_id in targets if node_id in cache}

    def get_step_results(self) -> list[StepResult]:
        return self._step_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_required(self, targets: list[str]) -> set[str]:
        """targets から逆向きに依存を辿り、実行が必要なノード id を返す。"""
        required: set[str] = set()
        stack = list(targets)
        while stack:
            node_id = stack.pop()
            if node_id in required:
                continue
            required.add(node_id)
            if node_id not in self._node_map:
                # input_dfs に直接ある場合はスキップ
                continue
            node = self._node_map[node_id]
            for dep in node.from_nodes:
                stack.append(dep)
        return required

    def _resolve_auto_from(self, required: set[str]) -> None:
        """from_nodes が空のノードに「直前ノード」を自動設定する。

        定義順で走査し、from_nodes が空の変換ノードには
        直前に出現した「Input Node または required ノード」を設定する。
        required 外のノードが挟まっても prev_id は維持する（Input Node は常に有効）。
        """
        prev_id: str | None = None
        for node in self._nodes:
            if node.is_input:
                # Input Node は常に直前候補として保持する
                prev_id = node.id
                continue
            if node.id not in required:
                # required 外の変換ノードはスキップするが prev_id はリセットしない
                continue
            if len(node.from_nodes) == 0 and prev_id is not None:
                node.from_nodes = [prev_id]
            prev_id = node.id

    def _topological_sort(self, required: set[str]) -> list[str]:
        """required ノードをトポロジカル順にソートして返す。"""
        visited: set[str] = set()
        result: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            if node_id in self._node_map:
                for dep in self._node_map[node_id].from_nodes:
                    if dep in required:
                        visit(dep)
            result.append(node_id)

        for node_id in required:
            visit(node_id)

        return result

    def _parse_resolver_cfg(
        self,
        node: Node,
        cache: dict[str, pl.DataFrame],
    ) -> tuple[str, str, dict[str, object]]:
        """Node の resolver_cfg からResolver名・Method・kwargs を取り出す。

        resolver_cfg は 1 キーのみ持つことを前提とする。
        例: {"polars": {"method": "select_columns", "columns": [...]}}
        """
        cfg = node.resolver_cfg
        if not cfg:
            return "unknown", "unknown", {}

        resolver_name = next(iter(cfg))
        method_cfg = dict(cfg[resolver_name])
        method = method_cfg.pop("method", "")

        # join の場合は複数 from_nodes の DataFrame をリストで渡す
        if method == "join" and len(node.from_nodes) > 1:
            method_cfg["dfs"] = [cache[nid] for nid in node.from_nodes]

        # output の場合は output_dir / node_id / cv_splits を追加
        if resolver_name == "output":
            method_cfg["output_dir"] = self._output_dir
            method_cfg["node_id"] = node.id
            if "cv" not in method_cfg:
                method_cfg["cv"] = self._cv_splits is not None
            if method_cfg.get("cv"):
                method_cfg["splits"] = self._cv_splits

        return resolver_name, str(method), method_cfg
