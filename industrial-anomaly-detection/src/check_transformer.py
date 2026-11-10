import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, load_config
from src.models.transformer import ForecastTransformer
from src.window_dataset import ForecastWindowDataset


def main():
    config = load_config()
    torch.manual_seed(config["experiment"]["seeds"][0])

    sequence_length = config["experiment"]["sequence_length"]

    dataset = ForecastWindowDataset(
        PROJECT_ROOT / "data/processed/hai/train_scaled.npz",
        sequence_length=sequence_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    history, target = next(iter(loader))

    model = ForecastTransformer(
        num_features=history.shape[-1],
        sequence_length=sequence_length,
    )

    prediction = model(history)
    loss = nn.MSELoss()(prediction, target)
    loss.backward()

    assert prediction.shape == target.shape
    assert torch.isfinite(loss)
    assert model.input_projection.weight.grad is not None
    assert torch.isfinite(model.input_projection.weight.grad).all()

    print("Input:", tuple(history.shape))
    print("Prediction:", tuple(prediction.shape))
    print("Target:", tuple(target.shape))
    print("Initial loss:", loss.item())
    print("Forward/backward: OK")


if __name__ == "__main__":
    main()