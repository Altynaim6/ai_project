import platform
import sys
from pathlib import Path
import argparse

import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from src.config import load_config, prepare_directories
from src.inspect_dataset import inspect_dataset


PROJECT_ROOT = Path(__file__).resolve().parent


def check_environment():
    print("Industrial Anomaly Detection")
    print("Python:", platform.python_version())
    print("Interpreter:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("PyG:", torch_geometric.__version__)

    # Проверяем вычисление и обратное распространение через GNN.
    torch.manual_seed(42)
    x = torch.randn(3, 4)
    edges = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    layer = GCNConv(4, 8)
    output = layer(x, edges)
    output.square().mean().backward()

    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()
    assert layer.lin.weight.grad is not None

    print("GNN forward/backward: OK")

    data_dir = PROJECT_ROOT / "data" / "raw" / "swat"
    data_dir.mkdir(parents=True, exist_ok=True)
    print("SWaT directory:", data_dir)
    print("Environment check completed")


def main():
    parser = argparse.ArgumentParser(
        description="Industrial anomaly detection project"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "check",
        help="Check configuration and environment",
    )

    inspect_command = commands.add_parser(
        "inspect",
        help="Show dataset columns and sample rows",
    )
    inspect_command.add_argument("path", type=Path)
    inspect_command.add_argument("--sheet", type=int, default=0)
    inspect_command.add_argument("--sep", default=",")

    args = parser.parse_args()

    try:
        config = load_config()
        prepare_directories(config)

        if args.command == "check":
            print("Project:", config["project_name"])
            print("Sequence length:", config["experiment"]["sequence_length"])
            print("Seeds:", config["experiment"]["seeds"])
            check_environment()

        elif args.command == "inspect":
            inspect_dataset(args.path, args.sheet, args.sep)

    except Exception as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()