import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT


def main():
    source = PROJECT_ROOT / "data/raw/hai/train1.csv"
    output_dir = PROJECT_ROOT / "results/hai/transformer_smoke/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = ["P4_ST_FD", "P4_ST_PS", "P4_ST_PO"]

    frame = pd.read_csv(
        source,
        sep=";",
        usecols=["time", "attack", *channels],
    )
    frame["time"] = pd.to_datetime(
        frame["time"],
        format="%Y-%m-%d %H:%M:%S",
    )

    peak_time = pd.Timestamp("2019-09-15 07:00:00")
    nearby = frame.loc[
        frame["time"].between(
            peak_time - pd.Timedelta(minutes=2),
            peak_time + pd.Timedelta(minutes=2),
        )
    ].copy()

    if nearby.empty:
        raise ValueError("No observations around the peak")

    print("Observations near the peak:")
    print(
        nearby.loc[
            nearby["time"].between(
                peak_time - pd.Timedelta(seconds=5),
                peak_time + pd.Timedelta(seconds=5),
            )
        ].to_string(index=False)
    )

    boundary = int(len(frame) * 0.8)
    train = frame.iloc[:boundary]
    validation = frame.iloc[boundary:]

    for name in channels:
        print(f"\n{name}")
        print("Train:")
        print(
            train[name].describe(
                percentiles=[0.01, 0.5, 0.99]
            ).to_string()
        )
        print("Validation:")
        print(
            validation[name].describe(
                percentiles=[0.01, 0.5, 0.99]
            ).to_string()
        )

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for axis, name in zip(axes, channels):
        axis.plot(nearby["time"], nearby[name], marker=".", markersize=3)
        axis.axvline(peak_time, color="red", linestyle="--")
        axis.set_ylabel(name)
        axis.grid(alpha=0.2)

    axes[0].set_title("Raw signals around the validation score peak")
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()

    output = output_dir / "peak_raw_signals.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print("\nSaved:", output)


if __name__ == "__main__":
    main()