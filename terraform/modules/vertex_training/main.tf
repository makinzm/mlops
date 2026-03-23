# ───────────────────────────────────────────────
# GCS バケット — データ・モデルのステージング用
# ───────────────────────────────────────────────
resource "google_storage_bucket" "staging" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = false  # 誤削除防止

  uniform_bucket_level_access = true

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 90 }  # 90日後に自動削除
  }
}

# ───────────────────────────────────────────────
# Artifact Registry — Docker イメージ保管
# ───────────────────────────────────────────────
resource "google_artifact_registry_repository" "mlops" {
  location      = var.region
  repository_id = "mlops"
  description   = "MLOps training container images"
  format        = "DOCKER"
}

# ───────────────────────────────────────────────
# サービスアカウント — Vertex AI 学習実行用
# 最小権限: GCS読み書き + ジョブ操作 + ログ書き込み + Docker pull
# ───────────────────────────────────────────────
resource "google_service_account" "training_sa" {
  account_id   = "training-sa"
  display_name = "Vertex AI Training Service Account"
  description  = "Vertex AI カスタムジョブ実行用（最小権限）"
}

resource "google_project_iam_member" "training_sa_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.training_sa.email}"
}

resource "google_project_iam_member" "training_sa_aiplatform" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.training_sa.email}"
}

resource "google_project_iam_member" "training_sa_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.training_sa.email}"
}

resource "google_project_iam_member" "training_sa_registry" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.training_sa.email}"
}

# ───────────────────────────────────────────────
# サービスアカウント — Kaggle Notebook / 推論読み取り専用
# 最小権限: GCS 読み取りのみ（書き込み不可）
# ───────────────────────────────────────────────
resource "google_service_account" "reader_sa" {
  account_id   = "reader-sa"
  display_name = "MLOps Reader Service Account"
  description  = "Kaggle Notebook / ローカル推論からモデルを読むための読み取り専用 SA"
}

resource "google_project_iam_member" "reader_sa_storage" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.reader_sa.email}"
}

# ───────────────────────────────────────────────
# 予算アラート — Terraform 管理外
# ───────────────────────────────────────────────
# Billing Budget の Terraform 作成には billing account レベルの
# roles/billing.budgets.admin が必要だが、個人プロジェクトでは
# 付与が困難（API が 400 invalid argument を返す既知の問題）。
#
# 代わりに GCP コンソールから手動で設定する:
#   コンソール → お支払い → 予算とアラート → 予算を作成
#
# 手順の詳細: docs/manual/1002_cost-monitoring.md
