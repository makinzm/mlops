.PHONY: help install test cov lint format typecheck clean
.PHONY: dataset train check serve inference submit

DEVBOX := devbox run --

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

# I/F: CreateDatasetInput(raw_name, output_name) -> CreateDatasetOutput(path, num_samples, num_features)
dataset:
	$(DEVBOX) uv run python -m src.cli dataset --raw=$(RAW) --output=$(OUT)

# I/F: TrainInput(features_name, model_name, params) -> TrainOutput(model_path, metrics, run_id)
train:
	$(DEVBOX) uv run python -m src.cli train --features=$(FEATURES) --model=$(MODEL) $(if $(PARAMS),--params="$(PARAMS)",)

# I/F: EvaluateInput(model_name, features_name) -> EvaluateOutput(metrics, details)
check:
	$(DEVBOX) uv run python -m src.cli check --model=$(MODEL) --features=$(FEATURES)

# I/F: ServeInput(model_name) -> ServeOutput(artifact_path, serving_info)
serve:
	$(DEVBOX) uv run python -m src.cli serve --model=$(MODEL)

# I/F: InferenceInput(model_name, features_name, output_name) -> InferenceOutput(predictions_path, num_predictions)
inference:
	$(DEVBOX) uv run python -m src.cli inference --model=$(MODEL) --features=$(FEATURES) --output=$(OUT)

# I/F: SubmitInput(predictions_path, submission_name) -> SubmitOutput(submission_id, status)
submit:
	$(DEVBOX) uv run python -m src.cli submit --predictions=$(PRED) --name=$(NAME)
