"""CLI entry point for ML workflow commands.

Usage:
    python -m src.cli <command> [options]

Commands:
    dataset    Create dataset from raw data
    train      Train model
    check      Evaluate model
    serve      Prepare model for serving
    inference  Run inference
    submit     Submit predictions
"""

import argparse
import sys


def cmd_dataset(args: argparse.Namespace) -> None:
    """Create dataset: raw -> features"""
    print(f"Creating dataset: raw={args.raw} -> output={args.output}")
    # TODO: Wire up CreateDatasetUseCase with adapters
    raise NotImplementedError("Implement dataset creation")


def cmd_train(args: argparse.Namespace) -> None:
    """Train model"""
    print(f"Training: features={args.features}, model={args.model}")
    # TODO: Wire up TrainUseCase with adapters
    raise NotImplementedError("Implement training")


def cmd_check(args: argparse.Namespace) -> None:
    """Evaluate model"""
    print(f"Evaluating: model={args.model}, features={args.features}")
    # TODO: Wire up EvaluateUseCase with adapters
    raise NotImplementedError("Implement evaluation")


def cmd_serve(args: argparse.Namespace) -> None:
    """Prepare model for serving"""
    print(f"Preparing for serving: model={args.model}")
    # TODO: Wire up ServeUseCase with adapters
    raise NotImplementedError("Implement serving")


def cmd_inference(args: argparse.Namespace) -> None:
    """Run inference"""
    print(f"Running inference: model={args.model}, features={args.features} -> {args.output}")
    # TODO: Wire up InferenceUseCase with adapters
    raise NotImplementedError("Implement inference")


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit predictions"""
    print(f"Submitting: predictions={args.predictions}, name={args.name}")
    # TODO: Wire up SubmitUseCase with adapters
    raise NotImplementedError("Implement submission")


def main() -> int:
    parser = argparse.ArgumentParser(description="ML workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # dataset
    p_dataset = subparsers.add_parser("dataset", help="Create dataset")
    p_dataset.add_argument("--raw", required=True, help="Raw data name")
    p_dataset.add_argument("--output", required=True, help="Output features name")
    p_dataset.set_defaults(func=cmd_dataset)

    # train
    p_train = subparsers.add_parser("train", help="Train model")
    p_train.add_argument("--features", required=True, help="Features data name")
    p_train.add_argument("--model", required=True, help="Model name to save")
    p_train.add_argument("--params", help="Hyperparameters (key=value,...)")
    p_train.set_defaults(func=cmd_train)

    # check
    p_check = subparsers.add_parser("check", help="Evaluate model")
    p_check.add_argument("--model", required=True, help="Model name")
    p_check.add_argument("--features", required=True, help="Features data name")
    p_check.set_defaults(func=cmd_check)

    # serve
    p_serve = subparsers.add_parser("serve", help="Prepare for serving")
    p_serve.add_argument("--model", required=True, help="Model name")
    p_serve.set_defaults(func=cmd_serve)

    # inference
    p_inference = subparsers.add_parser("inference", help="Run inference")
    p_inference.add_argument("--model", required=True, help="Model name")
    p_inference.add_argument("--features", required=True, help="Features data name")
    p_inference.add_argument("--output", required=True, help="Output name")
    p_inference.set_defaults(func=cmd_inference)

    # submit
    p_submit = subparsers.add_parser("submit", help="Submit predictions")
    p_submit.add_argument("--predictions", required=True, help="Predictions file path")
    p_submit.add_argument("--name", required=True, help="Submission name")
    p_submit.set_defaults(func=cmd_submit)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
