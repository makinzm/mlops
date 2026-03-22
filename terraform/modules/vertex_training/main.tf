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

# Cloud Billing が Pub/Sub に publish できるよう IAM を付与
resource "google_pubsub_topic_iam_member" "billing_pubsub" {
  topic  = google_pubsub_topic.budget_alerts.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:billing-export@system.gserviceaccount.com"
}

# ───────────────────────────────────────────────
# 予算アラート
# ───────────────────────────────────────────────
resource "google_billing_budget" "monthly_budget" {
  billing_account = var.billing_account_id
  display_name    = "MLOps Monthly Budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
    services = [
      "services/aiplatform.googleapis.com",   # Vertex AI
      "services/storage.googleapis.com",       # GCS
      "services/artifactregistry.googleapis.com",
    ]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(floor(var.budget_amount))
    }
  }

  # 50% / 80% / 100% / 120% で通知
  threshold_rules {
    threshold_percent = 0.5
    spend_basis       = "CURRENT_SPEND"
  }
  threshold_rules {
    threshold_percent = 0.8
    spend_basis       = "CURRENT_SPEND"
  }
  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "CURRENT_SPEND"
  }
  threshold_rules {
    threshold_percent = 1.2
    spend_basis       = "CURRENT_SPEND"
  }

  all_updates_rule {
    monitoring_notification_channels = []
    pubsub_topic                     = google_pubsub_topic.budget_alerts.id
    # メール通知はコンソールの請求設定で追加（Terraform 管理外）
  }
}

# ───────────────────────────────────────────────
# Cloud Function (Gen2) — 予算超過時ジョブ停止
# budget_action = "stop" の場合のみ実際に停止する
# budget_action = "warn" の場合は通知のみ（停止なし）
# ───────────────────────────────────────────────
resource "google_storage_bucket_object" "budget_enforcer_zip" {
  name   = "functions/budget_enforcer.zip"
  bucket = google_storage_bucket.staging.name
  source = "${path.module}/budget_enforcer.zip"
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
