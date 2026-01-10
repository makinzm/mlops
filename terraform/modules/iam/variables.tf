variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev/prod)"
  type        = string
}

variable "github_repository" {
  description = "GitHub repository (owner/repo format)"
  type        = string
}
