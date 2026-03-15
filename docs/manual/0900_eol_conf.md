# 0900: Competition Config のアーカイブ（EOL）

コンペ終了後に conf を `archive/` へ移動する手順。

---

## アーカイブ手順

```bash
mkdir -p conf/archive/2026/Q1/
git mv conf/competition/titanic conf/archive/2026/Q1/titanic
git commit -m "archive: titanic"
```

その後 `conf/config.yaml` の `competition:` を次のコンペに切り替える。

```yaml
# conf/config.yaml
defaults:
  - competition: house-prices/competition
  - _self_
```

---

## アーカイブ後のディレクトリ構造

```
conf/
  competition/
    {next-name}/
      competition.yaml
      eda.yaml
      preprocess/
        base.yaml

  archive/
    2026/Q1/
      titanic/              # git mv で移動済み
        competition.yaml
        eda.yaml
        preprocess/
          base.yaml
```
