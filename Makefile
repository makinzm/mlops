.PHONY: help install test cov lint format typecheck clean
.PHONY: dataset train check serve inference submit
.PHONY: tf-init tf-plan tf-apply tf-destroy
.PHONY: docker-build docker-push
.PHONY: kaggle-push-dataset kaggle-push-model kaggle-submit
.PHONY: gcp-train

DEVBOX := devbox run --
ENV ?= dev
TAG ?= latest

# ==============================================================================
# Development
# ==============================================================================
help:
	@echo "Development:"
	@echo "  make install      - Install dependencies (devbox shell)"
	@echo "  make test         - Run tests"
	@echo "  make cov          - Run tests with coverage"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Format code"
	@echo "  make typecheck    - Run type checker"
	@echo ""
	@echo "Workflow:"
	@echo "  make dataset      - Create dataset (raw -> features)"
	@echo "  make train        - Train model"
	@echo "  make check        - Evaluate model"
	@echo "  make serve        - Prepare model for serving"
	@echo "  make inference    - Run inference"
	@echo "  make submit       - Submit predictions"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make tf-init      - Initialize Terraform"
	@echo "  make tf-plan      - Plan Terraform changes"
	@echo "  make tf-apply     - Apply Terraform changes"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build training image"
	@echo "  make docker-push  - Push to Artifact Registry"
	@echo ""
	@echo "Kaggle:"
	@echo "  make kaggle-push-dataset - Push code to Kaggle Dataset"
	@echo "  make kaggle-push-model   - Push model to Kaggle Models"
	@echo "  make kaggle-submit       - Submit to competition"
	@echo ""
	@echo "GCP:"
	@echo "  make gcp-train    - Submit Vertex AI training job"

install:
	devbox shell

test:
	$(DEVBOX) uv run pytest

cov:
	$(DEVBOX) uv run pytest --cov=src --cov-report=term-missing

cov-html:
	$(DEVBOX) uv run pytest --cov=src --cov-report=html
	@echo "Open htmlcov/index.html"

lint:
	$(DEVBOX) uv run ruff check src tests

format:
	$(DEVBOX) uv run ruff format src tests

typecheck:
	$(DEVBOX) uv run mypy src

clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
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
# Docker
# ==============================================================================
docker-build:
	docker build -t mlops-train:$(TAG) -f docker/Dockerfile.train .

docker-push:
	docker tag mlops-train:$(TAG) $(REGISTRY)/mlops-training/train:$(TAG)
	docker push $(REGISTRY)/mlops-training/train:$(TAG)

# ==============================================================================
# Kaggle
# ==============================================================================
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

# ==============================================================================
# GCP
# ==============================================================================
gcp-train:
	gcloud ai custom-jobs create \
		--region=$(GCP_REGION) \
		--display-name=train-$(MODEL) \
		--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=$(REGISTRY)/mlops-training/train:$(TAG)
