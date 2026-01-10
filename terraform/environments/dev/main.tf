module "storage" {
  source = "../../modules/storage"

  project_id              = var.project_id
  region                  = var.region
  environment             = "dev"
  artifact_retention_days = 30
}

module "artifact_registry" {
  source = "../../modules/artifact_registry"

  project_id  = var.project_id
  region      = var.region
  environment = "dev"
}

module "iam" {
  source = "../../modules/iam"

  project_id        = var.project_id
  environment       = "dev"
  github_repository = var.github_repository
}
