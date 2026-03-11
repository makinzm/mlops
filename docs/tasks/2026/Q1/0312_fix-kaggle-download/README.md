# Fix: Kaggle Download unzip / force チェック

## 概要

`KaggleDownloader` に存在する2つのバグを修正する。

---

## バグ詳細

### Bug 1: `unzip=True` でも展開されない

**原因（2点）:**
1. `competition_download_files` は `unzip` パラメータを持たない（Kaggle API v1.8.3 仕様）。
   competition モードでは ZIP が展開されずに残る。
2. `dataset_download_files(unzip=True)` は「ダウンロードが発生したとき」のみ展開する。
   既存 ZIP がある場合はダウンロードをスキップするため、展開もスキップされる。

**修正方針:**
- `dataset_download_files` には引き続き `unzip=False` を渡す（API の unzip に依存しない）
- ダウンロード後、`cfg.unzip=True` のとき `_unzip_zips(output_dir)` で手動展開する
- dataset / competition 両モードで同じ処理パスを通すことで動作を統一する

### Bug 2: `force=False` のとき既存データがあってもエラーにならない

**原因:**
- 既存データのチェックロジックがない。Kaggle API が静かにスキップするだけ。
- ユーザーが意図せずデータを上書き（または上書きされないまま古いデータを使い続ける）リスクがある。

**修正方針:**
- `_check_force(output_dir)` メソッドを追加
- `force=False` のときにデータファイル（`.gitkeep` 等を除く）が存在すれば `FileExistsError` を送出
- dataset / competition 両モードでダウンロード前にチェックする

---

## 実装ファイル

- `src/infrastructure/downloader/kaggle.py` — メイン修正
- `tests/infrastructure/downloader/test_kaggle.py` — テスト追加

---

## 変更仕様

### `KaggleDownloader` に追加するメソッド

```python
def _check_force(self, output_dir: Path) -> None:
    """force=False かつデータファイルが存在する場合は FileExistsError を送出。"""

def _unzip_zips(self, output_dir: Path) -> None:
    """output_dir 内の ZIP ファイルをすべて展開し、ZIP を削除する。"""
```

### `_download_dataset` の変更

```
Before: api.dataset_download_files(..., unzip=cfg.unzip, force=cfg.force)
After:
  1. _check_force(output_dir)
  2. api.dataset_download_files(..., unzip=False, force=cfg.force)
  3. if cfg.unzip: _unzip_zips(output_dir)
```

### `_download_competition` の変更

```
Before: api.competition_download_files(..., force=cfg.force)
After:
  1. _check_force(output_dir)
  2. api.competition_download_files(..., force=cfg.force)
  3. if cfg.unzip: _unzip_zips(output_dir)
```

---

## テストケース追加

| テスト | 内容 |
|--------|------|
| `test_competition_unzips_when_unzip_true` | competition モードで ZIP が展開されること |
| `test_competition_does_not_unzip_when_unzip_false` | competition モードで `unzip=False` のとき ZIP が展開されないこと |
| `test_dataset_unzips_when_unzip_true` | dataset モードで ZIP が展開されること |
| `test_dataset_does_not_unzip_when_unzip_false` | dataset モードで `unzip=False` のとき ZIP が展開されないこと |
| `test_download_fails_when_files_exist_and_force_false` | dataset/competition で既存ファイルあり・`force=False` → `FileExistsError` |
| `test_download_succeeds_when_files_exist_and_force_true` | `force=True` なら既存ファイルがあってもダウンロード成功 |
| `test_download_succeeds_when_only_management_files_exist` | `.gitkeep` のみなら `force=False` でも成功 |

---

## 作業ステップ

- [x] feature branch 作成 (`fix/kaggle-download-unzip-force`)
- [ ] lefthook / CI 確認
- [ ] テスト追加（RED）→ commit --no-verify → TEST_LOG 保存
- [ ] 実装（GREEN）→ commit
- [ ] PR 作成
