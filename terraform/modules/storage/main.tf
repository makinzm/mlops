resource "google_storage_bucket" "mlops" {
  name                        = "${var.project_id}-mlops-${var.environment}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.environment == "dev"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = var.artifact_retention_days
    }
    action {
      type = "Delete"
    }
  }
}

# Folder structure markers (optional, GCS uses prefixes)
resource "google_storage_bucket_object" "folders" {
  for_each = toset(["data/raw/", "data/processed/", "data/features/", "models/", "mlflow/"])

  name    = each.value
  content = ""
  bucket  = google_storage_bucket.mlops.name
}
