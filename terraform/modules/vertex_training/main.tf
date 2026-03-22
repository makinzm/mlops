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
# Pub/Sub トピック — 予算アラート通知用
# ───────────────────────────────────────────────
resource "google_pubsub_topic" "budget_alerts" {
  name = "budget-alerts"
}

# ───────────────────────────────────────────────
# 予算アラート
# ───────────────────────────────────────────────
# プロジェクト番号を取得（Budget API は projects/NUMBER 形式が必要）
data "google_project" "current" {
  project_id = var.project_id
}

data "google_billing_account" "account" {
  count           = var.enable_budget ? 1 : 0
  billing_account = var.billing_account_id
}

resource "google_billing_budget" "monthly_budget" {
  count           = var.enable_budget ? 1 : 0
  billing_account = data.google_billing_account.account[0].id
  display_name    = "MLOps Monthly Budget"

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.8
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  # NOTE:
  # - budget_filter を省略するとこの billing account の全支出を監視する
  # - Pub/Sub 連携は Budget 作成後にコンソールから設定する
  #   (空の all_updates_rule {} は 400 エラーになる既知の問題)
  #   ref: https://github.com/hashicorp/terraform-provider-google/issues/9375
}

# ───────────────────────────────────────────────
# Cloud Function (Gen2) — 予算超過時ジョブ停止
# budget_action = "stop" の場合のみ実際に停止する
# budget_action = "warn" の場合は通知のみ（停止なし）
# ───────────────────────────────────────────────
data "archive_file" "budget_enforcer" {
  type        = "zip"
  source_dir  = "${path.module}/budget_enforcer"
  output_path = "${path.module}/budget_enforcer.zip"
}

resource "google_storage_bucket_object" "budget_enforcer_zip" {
  name   = "functions/budget_enforcer.zip"
  bucket = google_storage_bucket.staging.name
  source = data.archive_file.budget_enforcer.output_path
}

resource "google_cloudfunctions2_function" "budget_enforcer" {
  name        = "budget-enforcer"
  location    = var.region
  description = "予算超過時に Vertex AI ジョブを停止する（budget_action=stop の場合のみ）"

  build_config {
    runtime     = "python312"
    entry_point = "handle_budget_alert"
    source {
      storage_source {
        bucket = google_storage_bucket.staging.name
        object = google_storage_bucket_object.budget_enforcer_zip.name
      }
    }
  }

  service_config {
    environment_variables = {
      BUDGET_ACTION = var.budget_action
      GCP_PROJECT   = var.project_id
      GCP_REGION    = var.region
    }
    service_account_email = google_service_account.training_sa.email
  }

  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.budget_alerts.id
  }
}
