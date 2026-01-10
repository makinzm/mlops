resource "google_artifact_registry_repository" "training" {
  repository_id = "mlops-training"
  location      = var.region
  format        = "DOCKER"
  description   = "Training container images for MLOps"

  labels = {
    environment = var.environment
  }
}
