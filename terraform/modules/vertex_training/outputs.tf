output "bucket_name" {
  value = google_storage_bucket.staging.name
}

output "container_registry_uri" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mlops.repository_id}"
}

output "training_sa_email" {
  value = google_service_account.training_sa.email
}

output "reader_sa_email" {
  value = google_service_account.reader_sa.email
}
