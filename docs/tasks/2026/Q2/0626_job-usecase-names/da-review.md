## 🔍 DAレビュー - Round 1

**レビュー対象**: `refactor/job-usecase-names` (main...refactor/job-usecase-names, commits 2726449 / 47d6df1)
**レビュー日時**: 2026-06-26

### 指摘事項

#### [重要度: 高] 旧 `conf/cloud/vertex.yaml` が削除されず孤児ファイルとして残存
- **該当箇所**: `conf/cloud/vertex.yaml`（git にトラッキングされたまま）
- **問題**: 本リファクタは `conf/cloud/vertex.yaml` → `conf/infra/vertex.yaml` への移行だが、新しい `conf/infra/vertex.yaml` が「追加」されただけで、旧ファイルが `git rm` されていない。diff は move/rename ではなく copy になっており、`conf/cloud/vertex.yaml`（先頭キー `cloud:`）がそのままリポジトリに残っている。
  - 確認コマンド: `git ls-files conf/cloud conf/infra` → 両方が出力される
  - 旧ファイルの中身は `cloud:` namespace を定義しており、まさに今回のリファクタが除去対象としたもの
- **理由**: (1) リファクタの目的（"cloud" 除去）に真っ向から反する死んだ設定ファイルが残る。(2) `conf/config.yaml` の struct には `cloud:` キーが存在しなくなったため、誰かが誤って `usecase` config の defaults を `- /cloud: vertex` に戻すと無言で `cloud:` グループに解決され混乱を招く。(3) 将来の読者が「infra と cloud のどちらが正か」を判別できない。
- **提案**: `git rm conf/cloud/vertex.yaml` を実行し、空になった `conf/cloud/` ディレクトリも除去する。グレップで `/cloud` 参照が他に無いことは確認済み（残るのはこの孤児ファイルの `cloud:` 行のみ）。

#### [重要度: 中] README 計画書が実装と矛盾（`cloud_job_name` の扱い）
- **該当箇所**: `docs/tasks/2026/Q2/0626_job-usecase-names/README.md:38`
- **問題**: README は「`src/domain/data/job_manifest.py`: `cloud_job_name` フィールドはすでにドメイン概念なので変更なし」と記載しているが、実装では `cloud_job_name` → `job_name` にリネームされている（`src/domain/data/job_manifest.py:44` および `job_submit.py` の diff で確認）。タスク説明とも一致する（実装側が正しい）。さらに README の config セクション（30-35行）には `conf/cloud/vertex.yaml` → `conf/infra/vertex.yaml` の移動が一切記載されていない。
- **理由**: DoD「なぜそのテストが必要かあとから読んでもわかるように」「実行計画を README に書く」に照らすと、計画書が実装と乖離していると後続レビュワー・将来の自分が追跡できない。上記「高」の孤児ファイル漏れも、計画に conf 移動が書かれていなかったことが一因と推測される。
- **提案**: README:38 を実態（`cloud_job_name` → `job_name` にリネーム。domain の name_deny には "cloud" は無いが、インフラ固有語を排しドメイン語へ統一するため変更）に修正し、`conf/cloud` → `conf/infra` の移動を config セクションに追記する。

#### [重要度: 中] presentation 層の `cloud_config.py` / `ensure_cloud_config` が新 `infra` 抽象と命名不一致
- **該当箇所**: `src/presentation/cloud_config.py:16` (`ensure_cloud_config`)、ファイル名 `cloud_config.py`、`src/presentation/runners.py:16-17,185,220,252`
- **問題**: presentation 層（合成ルート / DI）はインフラを知ってよい層なので "cloud" 語の残存自体は許容範囲。ただし本関数は今や `conf/infra/vertex.yaml` をロードし `cfg.infra` を解決する。設定キー・config グループは `infra` に統一されたのに、ヘルパー名・ファイル名だけ `cloud_config` / `ensure_cloud_config` のまま残り、読み手に「cloud と infra のどちらが現行語か」の認知負荷を与える。docstring（1-4,17,20行）も `cloud 設定` のまま。
- **理由**: 今回のリファクタの主旨は語彙の統一。presentation を対象外と割り切るなら割り切るで一貫させるべきだが、中身は `infra` を扱うのに名前だけ `cloud` という中途半端な状態は保守性を下げる。
- **提案**: `cloud_config.py` → `infra_config.py`、`ensure_cloud_config` → `ensure_infra_config` にリネームし docstring も `infra 設定` に更新する（presentation 層なので mille には抵触しないが語彙を揃える）。スコープを usecase/domain/conf に限定する判断ならそれでも可だが、その方針を README に明記すること。

#### [重要度: 低] テスト docstring に旧フィールド名が残る
- **該当箇所**: `tests/usecase/training/test_job_download_usecase.py:133`
- **問題**: docstring が `get_job_status が cloud_job_name で呼ばれること` のままだが、アサーションは `_FAKE_JOB_NAME`（= `job_name`）を使用している。振る舞いは正しいが説明が旧名。
- **提案**: docstring を `job_name で呼ばれること` に更新。

#### [重要度: 低] usecase docstring の説明語が「クラウド設定」のまま
- **該当箇所**: `src/usecase/training/job_submit.py`（`cfg.infra.*: クラウド設定` 等のコメント）、同 `job_train.py`
- **問題**: 識別子は `infra` に統一されたが、説明コメントが「クラウド設定」と特定インフラ前提の語で残る。mille は識別子を見るため抵触しないが、語彙統一の観点では `インフラ設定 / 実行環境設定` 等が望ましい。
- **提案**: 説明コメントを「インフラ設定」に揃える（任意・軽微）。

#### [重要度: 低] `JobManifest.load` の後方互換: 旧 manifest が無言で `job_name=""` になる
- **該当箇所**: `src/domain/data/job_manifest.py:104` (`data.get("job_name", "")`)
- **問題**: 旧名 `cloud_job_name` で保存済みの manifest を読むと `job_name` が空文字に落ちる。ただし保存先ディレクトリも `cloud_jobs_history` → `job_history` に変わっており、旧 manifest は旧ディレクトリに隔離されるため実害は限定的。
- **提案**: 対応不要（情報共有）。気になるなら `data.get("job_name", data.get("cloud_job_name", ""))` で1行の互換読みを入れてもよいが、ephemeral データなので過剰設計の可能性が高い。

### 良い点
- usecase 層・domain 層から `cloud` / `vertex` / `remote` が完全に排除されている（`grep -rni` で usecase 配下 0 件、`uv run mille check` で `src_usecase` 0 violations）。`mille.toml` の `src_usecase.name_deny` に `"cloud"` が追加され再発が機械的に防止されている。
- `JobManifest.cloud_job_name` → `job_name` のリネームが dataclass 定義・`save()`・`load()` の全箇所で一貫している。
- Hydra の config グループ解決が健全。`conf/usecase/job_*.yaml` が `- /infra: vertex` を defaults で参照し、実際に `compose(config_name='config', overrides=['usecase=job_train'])` で `cfg.infra.machine_type` / `cfg.job_history_dir` が正しく解決されることを確認した。
- 検証が clean: 関連テスト 77 件 pass、`uv run ruff check` All checks passed、`uv run mille check` 0 violations。
- TDD の痕跡が明確（RED コミット 2726449 → GREEN コミット 47d6df1、`--no-verify` 運用も caution.md 準拠）。

### 総評
リファクタの中核（usecase/domain/conf/test の語彙統一、mille ガード追加、フィールドリネームの一貫性、Hydra 解決の健全性）は丁寧に仕上がっており、テスト・lint・mille も clean。ただし **旧 `conf/cloud/vertex.yaml` の削除漏れ** が「高」として残っており、これは今回のリファクタの目的そのもの（cloud 除去）に反する死んだ設定ファイルなので、PR 前に必ず `git rm` すべき。加えて README 計画書が実装と矛盾している点（`cloud_job_name` の扱い・conf 移動の未記載）と、presentation 層の命名不一致が中程度の改善余地。

### 判定
- [ ] ✅ LGTM（問題なし、マージ可能）
- [x] 🔧 要修正（指摘対応後、再レビュー）
- [ ] 🚨 要相談（人間の判断が必要）

**必須対応（高）**: `git rm conf/cloud/vertex.yaml`（空ディレクトリも除去）。
**推奨対応（中）**: README:38 の修正と conf 移動の追記 / presentation 命名の統一（またはスコープ外方針の明記）。
低レベル指摘は任意。高の対応後に再レビューします。
