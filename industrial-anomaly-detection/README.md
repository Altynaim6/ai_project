# Industrial Cyber Physical Anomaly Detection

**Selected topic:** Cyber-Physical Anomaly Detection in Industrial Control Systems via Sensor-Actuator Sequence Mining  
**Student:** Altynai Munduzbaeva  
**Project number:** 16 — TODO: confirm project number against the specification reported as 17.

This Python research project studies inconsistencies between industrial sensor measurements and actuator states. The intended system processes signals, compares detection models, preserves prediction history and publishes warnings.

## Scope of the first commit

This commit publishes configuration, CSV inspection, HAI schema checks, preprocessing, chronological partitions, forecast windows and data-pipeline tests. Model training/evaluation, graph models, streaming and MQTT are added in later commits. The commands below do not require those later modules.

The complete local project has evaluated Transformer, Random Forest and standalone GCN runs. Its hybrid GCN + Transformer is implemented and smoke-tested; full hybrid training and evaluation remain pending. These model components are outside this first commit.

## Dataset and experimental limits

SWaT access has been requested. Development currently uses [HAI 20.07](https://github.com/icsdataset/hai), a publicly available ICS dataset from a testbed augmented with hardware-in-the-loop simulation. Its process and feature set differ from SWaT, so its preprocessing and graph must be defined independently. Findings apply to HAI only.

The selected schema has 59 channels: 55 numeric and four binary. Time and attack labels are excluded from features. The data have one-second observation intervals.

Forecast v2 uses normal train1 data and the normal portion of the first 70% of train2 for training. The next 15% is validation and the last 15% is separate calibration. Attack observations and 300-second guards are excluded from normal training. Windows never cross gaps, source boundaries or excluded intervals. Numeric scaling fits normal training data only and applies an engineering-range floor; binary inputs retain 0/1.

The manifests record historical stages. Some reservation records describe test2 as unopened at that earlier time. **Test2 has since been inspected and evaluated; it is not currently a fresh blind test.** A warning identifies an unusual observation, not proof of a cyberattack.

## Environment

Development uses Windows, PowerShell and Python 3.12.10. Keep using the existing `.venv` if it is already configured. For a new data-pipeline environment, run from `industrial-anomaly-detection`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn joblib pytest openpyxl
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Activation is optional because commands use the environment's Python directly. Select alternative PyTorch builds using the [official installation guide](https://pytorch.org/get-started/locally/).

`requirements.txt` is the existing full Windows development snapshot, including dependencies for later stages. It is not a minimal cross-platform dependency list. Its pinned CPU torch build uses the PyTorch wheel index. The minimal installation above does not claim exact reproduction of the full development environment.

## Input files and inspection

Obtain HAI 20.07 from its official repository and place development recordings at:

```text
data/raw/hai/train1.csv
data/raw/hai/train2.csv
```

Raw data and generated arrays are excluded from Git. The small synthetic fixture is included and can be inspected without downloading HAI:

```powershell
.\.venv\Scripts\python.exe src\inspect_dataset.py tests\fixtures\sample_sensor_data.csv
```

HAI uses a semicolon delimiter. After obtaining the files:

```powershell
.\.venv\Scripts\python.exe src\inspect_dataset.py data\raw\hai\train1.csv --sep ";"
.\.venv\Scripts\python.exe src\check_hai.py data\raw\hai\train1.csv
.\.venv\Scripts\python.exe src\check_hai.py data\raw\hai\train2.csv
```

Review timestamp order, continuity, missing/non-numeric/infinite values and attack-label validity before preparation.

## Prepare a new version

`prepare_v2` refuses to overwrite an existing output, artifact, report or split-manifest version. Historical manifests are retained in this repository, so running the default preparation version can raise `FileExistsError` even in a new clone.

Use an unused version name:

```powershell
.\.venv\Scripts\python.exe -c "from src.prepare_hai_v2 import prepare_v2; prepare_v2(version='hai_mixed_v2_local')"
```

Outputs:

```text
data/processed/hai_mixed_v2_local/                 # train / validation / calibration NPZ
artifacts/hai/preprocessing_hai_mixed_v2_local/    # scaler bundle and metadata
reports/hai_mixed_v2_local/                       # input audit and scaling summary
manifests/hai_mixed_v2_local_split.json           # preparation protocol
```

Choose another name if this version already exists. Do not delete historical records just to rerun preparation. Later training scripts must use the matching data/preprocessing version; they do not automatically switch to `_local` outputs.

## First-commit tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_hai_v2.py -q
```

These tests use synthetic data and check the scaling floor, attack guards, continuous windows and split isolation. They need no downloaded HAI data and do not measure detection quality. `test_hai_mixed.py` belongs to the next commit because it imports Transformer code.

## Repository structure

| Path | Purpose |
|---|---|
| `configs/base.json` | General defaults; its SWaT setting retains the original project target |
| `configs/hai_20_07_schema.json` | Selected HAI inputs and channel semantics |
| `src/` | Data inspection, checking, preparation and window datasets in this commit |
| `tests/test_hai_v2.py` | Independent v2 data-pipeline checks |
| `tests/fixtures/` | Small synthetic inspection example |
| `manifests/` | Historical split, reservation and overlap audit records |

Use the explicit HAI preparation command above rather than assuming `base.json` automatically selects HAI. `main.py` and model commands are added in later commits.

## Files kept outside Git

Ignore rules exclude environments, raw/prepared data, model artifacts, generated result histories, document outputs, temporary builds/tests, archives and downloaded tools. Selected final metrics, figures and reports can be curated for the documentation commit. Ignore rules do not remove already tracked files.

No public model-download URL or completed final hybrid benchmark is claimed in this commit. Follow the dataset source's terms when obtaining or redistributing data.
