terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project                     = var.project_id
  region                      = var.region
  user_project_override       = true
  billing_project             = var.project_id
}

module "vertex_training" {
  source = "./modules/vertex_training"

  project_id         = var.project_id
  region             = var.region
  bucket_name        = var.bucket_name
  enable_budget      = var.enable_budget
  budget_amount      = var.budget_amount
  budget_alert_email = var.budget_alert_email
  budget_action      = var.budget_action
  billing_account_id = var.billing_account_id
}
