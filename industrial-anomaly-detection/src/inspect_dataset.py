import argparse
from pathlib import Path

import pandas as pd


def inspect_dataset(path: Path, sheet: int, separator: str):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension == ".csv":
        frame = pd.read_csv(
            path,
            sep=separator,
            nrows=5,
            encoding="utf-8-sig",
        )
    elif extension == ".xlsx":
        with pd.ExcelFile(path) as workbook:
            print("Sheets:", workbook.sheet_names)
            frame = pd.read_excel(
                workbook,
                sheet_name=sheet,
                nrows=5,
            )
    else:
        raise ValueError("Supported formats: .csv, .xlsx")

    print("\nFile:", path.name)
    print("Preview columns:", len(frame.columns))

    print("\nColumn names:")
    for index, column in enumerate(frame.columns):
        print(f"{index}: {column!r}")

    print("\nFirst rows:")
    print(frame.to_string(index=False))

    print("\nPreview data types:")
    print(frame.dtypes.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Inspect dataset headers and sample rows."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--sheet", type=int, default=0)
    parser.add_argument("--sep", default=",")
    args = parser.parse_args()

    try:
        inspect_dataset(args.path, args.sheet, args.sep)
    except Exception as error:
        parser.exit(1, f"Inspection failed: {error}\n")


if __name__ == "__main__":
    main()