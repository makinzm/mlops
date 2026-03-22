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

variable "budget_amount" {
  description = "月間予算上限（USD）"
  type        = number
  default     = 10
}

variable "budget_alert_email" {
  description = "予算アラート通知先メールアドレス"
  type        = string
}

variable "budget_action" {
  description = "予算超過時のアクション: 'warn'（通知のみ）または 'stop'（ジョブ停止）"
  type        = string
  default     = "warn"
  validation {
    condition     = contains(["warn", "stop"], var.budget_action)
    error_message = "budget_action は 'warn' または 'stop' のみ指定できます。"
  }
}

variable "billing_account_id" {
  description = "GCP 請求アカウント ID（例: 012345-ABCDEF-789012）"
  type        = string
}
