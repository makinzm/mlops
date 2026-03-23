variable "project_id" {
  description = "GCP プロジェクト ID"
  type        = string
}

variable "region" {
  description = "GCP リージョン"
  type        = string
  default     = "asia-northeast1"
}

variable "bucket_name" {
  description = "GCS バケット名（グローバルで一意）"
  type        = string
}
