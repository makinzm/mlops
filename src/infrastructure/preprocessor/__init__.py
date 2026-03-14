"""
前処理インフラ層。

- registry.py    : Resolver の登録・graceful skip
- dag_runner.py  : DAG 依存解決・実行エンジン
- visualizer.py  : pipeline_dag.html 生成
- resolvers/     : PolarsResolver / SklearnResolver / OutputResolver
"""
