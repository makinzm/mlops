output "bucket_name" {
  value = module.storage.bucket_name
}

output "artifact_registry_url" {
  value = module.artifact_registry.repository_url
}

output "github_actions_sa" {
  value = module.iam.github_actions_sa_email
}

output "workload_identity_provider" {
  value = module.iam.workload_identity_provider
}
