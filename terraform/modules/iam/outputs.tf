output "vertex_training_sa_email" {
  description = "Vertex AI training service account email"
  value       = google_service_account.vertex_training.email
}

output "github_actions_sa_email" {
  description = "GitHub Actions service account email"
  value       = google_service_account.github_actions.email
}

output "workload_identity_provider" {
  description = "Workload Identity Provider for GitHub Actions"
  value       = google_iam_workload_identity_pool_provider.github.name
}
