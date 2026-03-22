output "staging_bucket_uri" {
  description = "conf/gcp/vertex.yaml の staging_bucket に設定する値"
  value       = "gs://${module.vertex_training.bucket_name}"
}

output "container_registry_uri" {
  description = "Docker イメージの push 先 URI（タグなし）"
  value       = module.vertex_training.container_registry_uri
}

output "training_service_account_email" {
  description = "conf/gcp/vertex.yaml の service_account に設定する値"
  value       = module.vertex_training.training_sa_email
}

output "reader_service_account_email" {
  description = "Kaggle Notebook から GCS モデルを読む SA"
  value       = module.vertex_training.reader_sa_email
}
