variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-northeast1"
}

variable "environment" {
  description = "Environment name (dev/prod)"
  type        = string
}

variable "artifact_retention_days" {
  description = "Days to retain artifacts before deletion"
  type        = number
  default     = 90
}
