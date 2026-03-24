"""
Resolver レジストリ。

RESOLVER_REGISTRY に登録されていない Resolver / Method への呼び出しは
例外を上げず StepResult(status="skipped") を返す（graceful skip）。
実行中に例外が発生した場合は StepResult(status="failed") を返し、後続ステップを継続する。
"""

import polars as pl

from src.domain.data.preprocessor import StepResult
from src.infrastructure.preprocessor.resolvers.image_resolver import ImageResolver
from src.infrastructure.preprocessor.resolvers.output_resolver import OutputResolver
from src.infrastructure.preprocessor.resolvers.polars_resolver import PolarsResolver
from src.infrastructure.preprocessor.resolvers.sklearn_resolver import SklearnResolver

# Resolver 名 → Resolver クラスのマッピング
# 新しい Resolver を追加する場合はここに登録する
ResolverType = (
    type[PolarsResolver] | type[SklearnResolver] | type[OutputResolver] | type[ImageResolver]
)

RESOLVER_REGISTRY: dict[str, ResolverType] = {
    "polars": PolarsResolver,
    "sklearn": SklearnResolver,
    "output": OutputResolver,
    "image": ImageResolver,
}


def run_step(
    df: pl.DataFrame,
    resolver_name: str,
    method: str,
    kwargs: dict[str, object],
) -> tuple[pl.DataFrame, StepResult]:
    """1ステップを実行し (結果DataFrame, StepResult) を返す。

    エラーが発生しても例外を上げず StepResult に記録する。
    パイプライン継続のため、エラー時は変換前の DataFrame を返す。
    """
    # Resolver 未登録
    if resolver_name not in RESOLVER_REGISTRY:
        return df, StepResult(
            resolver=resolver_name,
            method=method,
            status="skipped",
            reason="resolver not found",
        )

    resolver = RESOLVER_REGISTRY[resolver_name]()

    # Method 未実装
    if method not in resolver.supported_methods():
        return df, StepResult(
            resolver=resolver_name,
            method=method,
            status="skipped",
            reason="method not found in resolver",
        )

    # 実行
    try:
        result_df = resolver.execute(df, method, **kwargs)
        return result_df, StepResult(
            resolver=resolver_name,
            method=method,
            status="ok",
            reason=None,
        )
    except Exception as e:
        return df, StepResult(
            resolver=resolver_name,
            method=method,
            status="failed",
            reason=str(e),
        )
