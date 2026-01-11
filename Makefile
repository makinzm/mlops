.PHONY: help install sync test cov cov-xml lint format typecheck clean
.PHONY: dataset train check serve inference submit
.PHONY: tf-init tf-plan tf-apply tf-destroy
.PHONY: docker-build docker-push gcloud-build
.PHONY: kaggle-setup kaggle-push-dataset kaggle-push-model kaggle-submit kaggle-inference-submit
.PHONY: gcp-train gcp-download-model

# CI環境ではdevboxを使わない
DEVBOX := $(if $(CI),,devbox run --)
ENV ?= dev
TAG ?= latest
MODEL ?= latest
MSG ?= auto
EXTRA ?=

# ==============================================================================
# Development
# ==============================================================================
help:
	@echo "Development:"
	@echo "  make install       - Install dependencies (devbox shell)"
	@echo "  make sync          - Sync dependencies with uv (EXTRA=dev,kaggle,gcp)"
	@echo "  make test          - Run tests"
	@echo "  make cov           - Run tests with coverage"
	@echo "  make cov-xml       - Run tests with coverage (XML output for CI)"
	@echo "  make lint          - Run linter"
	@echo "  make format        - Format code"
	@echo "  make typecheck     - Run type checker"
	@echo ""
	@echo "Workflow:"
	@echo "  make dataset       - Create dataset (raw -> features)"
	@echo "  make train         - Train model"
	@echo "  make check         - Evaluate model"
	@echo "  make serve         - Prepare model for serving"
	@echo "  make inference     - Run inference"
	@echo "  make submit        - Submit predictions"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make tf-init       - Initialize Terraform"
	@echo "  make tf-plan       - Plan Terraform changes"
	@echo "  make tf-apply      - Apply Terraform changes"
	@echo ""
	@echo "Docker/GCloud:"
	@echo "  make docker-build  - Build training image locally"
	@echo "  make docker-push   - Push to Artifact Registry"
	@echo "  make gcloud-build  - Build and push via Cloud Build"
	@echo ""
	@echo "Kaggle:"
	@echo "  make kaggle-setup            - Configure Kaggle credentials"
	@echo "  make kaggle-push-dataset     - Push code to Kaggle Dataset"
	@echo "  make kaggle-push-model       - Push model to Kaggle Models"
	@echo "  make kaggle-submit           - Submit predictions to competition"
	@echo "  make kaggle-inference-submit - Run inference and submit"
	@echo ""
	@echo "GCP:"
	@echo "  make gcp-train          - Submit Vertex AI training job"
	@echo "  make gcp-download-model - Download model from GCS"

install:
	devbox shell

sync:
	$(DEVBOX) uv sync $(if $(EXTRA),--extra $(EXTRA),)

test:
	$(DEVBOX) uv run pytest

cov:
	$(DEVBOX) uv run pytest --cov=src --cov-report=term-missing

cov-xml:
	$(DEVBOX) uv run pytest --cov=src --cov-report=xml

cov-html:
	$(DEVBOX) uv run pytest --cov=src --cov-report=html
	@echo "Open htmlcov/index.html"

lint:
	$(DEVBOX) uv run ruff check src tests

format:
	$(DEVBOX) uv run ruff format src tests

typecheck:
	$(DEVBOX) uv run ty check

clean:
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ==============================================================================
# ML Workflow
# ==============================================================================
# Usage:
#   make dataset RAW=train OUT=train_features
#   make train FEATURES=train_features MODEL=baseline PARAMS="lr=0.01"
#   make check MODEL=baseline FEATURES=val_features
#   make serve MODEL=baseline
#   make inference MODEL=baseline FEATURES=test_features OUT=predictions
#   make submit PRED=outputs/predictions/predictions.csv NAME=submission_v1
# ==============================================================================

dataset:
	$(DEVBOX) uv run python -m src.cli dataset --raw=$(RAW) --output=$(OUT)

train:
	$(DEVBOX) uv run python -m src.cli train --features=$(FEATURES) --model=$(MODEL) $(if $(PARAMS),--params="$(PARAMS)",)

check:
	$(DEVBOX) uv run python -m src.cli check --model=$(MODEL) --features=$(FEATURES)

serve:
	$(DEVBOX) uv run python -m src.cli serve --model=$(MODEL)

inference:
	$(DEVBOX) uv run python -m src.cli inference --model=$(MODEL) --features=$(FEATURES) --output=$(OUT)

submit:
	$(DEVBOX) uv run python -m src.cli submit --predictions=$(PRED) --name=$(NAME)

# ==============================================================================
# Infrastructure (Terraform)
# ==============================================================================
tf-init:
	cd terraform/environments/$(ENV) && $(DEVBOX) terraform init

tf-plan:
	cd terraform/environments/$(ENV) && $(DEVBOX) terraform plan

tf-apply:
	cd terraform/environments/$(ENV) && $(DEVBOX) terraform apply

tf-destroy:
	cd terraform/environments/$(ENV) && $(DEVBOX) terraform destroy

# ==============================================================================
# Docker / Cloud Build
# ==============================================================================
docker-build:
	docker build -t mlops-train:$(TAG) -f docker/Dockerfile.train .

docker-push:
	docker tag mlops-train:$(TAG) $(REGISTRY)/mlops-training/train:$(TAG)
	docker push $(REGISTRY)/mlops-training/train:$(TAG)

gcloud-build:
	gcloud builds submit \
		--tag $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/mlops-training/train:$(TAG) \
		--file docker/Dockerfile.train \
		.

# ==============================================================================
# Kaggle
# ==============================================================================
kaggle-setup:
	@mkdir -p ~/.kaggle
	@test -n "$(KAGGLE_JSON)" && echo '$(KAGGLE_JSON)' > ~/.kaggle/kaggle.json || true
	@chmod 600 ~/.kaggle/kaggle.json 2>/dev/null || true

kaggle-push-dataset:
	$(DEVBOX) uv run python -c "\
		from pathlib import Path; \
		from src.adapters.kaggle import KaggleConfig, KaggleDatasetPusher; \
		import os; \
		config = KaggleConfig( \
			competition=os.environ.get('KAGGLE_COMPETITION', 'your-competition'), \
			username=os.environ.get('KAGGLE_USERNAME', 'your-username'), \
		); \
		pusher = KaggleDatasetPusher(config); \
		pusher.package_and_push([Path('src'), Path('configs')], '$(MSG)'); \
		print('Dataset pushed:', config.dataset_slug)"

kaggle-push-model:
	$(DEVBOX) uv run python -c "\
		from src.adapters.kaggle import KaggleConfig, KaggleModelStore; \
		import os; \
		config = KaggleConfig( \
			competition=os.environ.get('KAGGLE_COMPETITION', 'your-competition'), \
			username=os.environ.get('KAGGLE_USERNAME', 'your-username'), \
		); \
		store = KaggleModelStore(config); \
		store.push_to_kaggle_models('$(MODEL)', '$(MODEL)', '$(MSG)'); \
		print('Model pushed:', config.model_slug)"

kaggle-submit:
	$(DEVBOX) uv run python -c "\
		import pandas as pd; \
		from src.adapters.kaggle import KaggleConfig, KaggleServingGateway; \
		import os; \
		config = KaggleConfig( \
			competition=os.environ.get('KAGGLE_COMPETITION', 'your-competition'), \
			username=os.environ.get('KAGGLE_USERNAME', 'your-username'), \
		); \
		predictions = pd.read_csv('$(PRED)'); \
		gateway = KaggleServingGateway(config); \
		gateway.submit(predictions, '$(NAME)'); \
		print('Submitted:', '$(NAME)')"

kaggle-inference-submit:
	$(DEVBOX) uv run python -c "\
		import pandas as pd; \
		from src.adapters.kaggle import KaggleConfig, KaggleServingGateway; \
		import os; \
		predictions = pd.DataFrame({'id': [1, 2, 3], 'prediction': [0, 1, 0]}); \
		config = KaggleConfig( \
			competition=os.environ.get('KAGGLE_COMPETITION', 'your-competition'), \
			username=os.environ.get('KAGGLE_USERNAME', 'your-username'), \
		); \
		gateway = KaggleServingGateway(config); \
		gateway.submit(predictions, '$(MSG)'); \
		print('Submitted:', '$(MSG)')"

# ==============================================================================
# GCP
# ==============================================================================
gcp-train:
	gcloud ai custom-jobs create \
		--region=$(GCP_REGION) \
		--display-name=train-$(TAG) \
		--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=$(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/mlops-training/train:$(TAG)

gcp-download-model:
	mkdir -p outputs/models
	gsutil cp gs://$(GCP_PROJECT_ID)-mlops-dev/models/$(MODEL)/model.pkl outputs/models/$(MODEL).pkl
