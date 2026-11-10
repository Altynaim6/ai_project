"""Sequential replay with frozen v2 artifacts, causal windows and incremental CSV logs."""
import argparse
import csv
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT
from src.evaluate_transformer_mixed import load_frozen_run
from src.mixed_forecast_loss import mixed_forecast_scores
from src.train_transformer_mixed import file_hash, write_json


FIELDS = ["event_id", "timestamp", "status", "prediction", "score", "numeric_mse",
          "binary_ce", "threshold", "history_size", "processing_ms", "detail"]


class StreamingDetector:
    """Each push consumes one observation; no label is used by this interface."""
    def __init__(self, model, bundle, threshold, sequence_length):
        self.model = model.eval()
        self.bundle = bundle
        self.threshold = float(threshold)
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)
        self.last_time = None

    @classmethod
    def from_run(cls, model_dir):
        model, bundle, calibration, checkpoint = load_frozen_run(model_dir)
        if bundle.get("format_version") != 2:
            raise ValueError("This simulator requires v2 preprocessing")
        detector = cls(model, bundle, calibration["threshold"], checkpoint["model_config"]["sequence_length"])
        detector.artifact_metadata = {"best_epoch": checkpoint["epoch"],
                                      "training_status": calibration.get("training_status", "see training run"),
                                      "model_sha256": calibration["model_sha256"],
                                      "preprocessing_sha256": calibration["preprocessing_sha256"]}
        return detector

    def reset(self):
        self.buffer.clear()
        self.last_time = None

    def push(self, observation):
        started = time.perf_counter()
        result = dict(timestamp=str(observation.get("time", "")), status="INSUFFICIENT_DATA",
                      prediction=None, score=None, numeric_mse=None, binary_ce=None,
                      threshold=self.threshold, history_size=len(self.buffer), detail="")
        try:
            timestamp = datetime.strptime(observation["time"], "%Y-%m-%d %H:%M:%S")
            names = self.bundle["numeric_columns"]
            raw_numeric = [float(observation[name]) for name in names]
            raw_binary = [float(observation[name]) for name in self.bundle["binary_columns"]]
            if not all(math.isfinite(value) for value in raw_numeric + raw_binary):
                raise ValueError("Missing/non-finite feature")
            if any(value not in (0, 1) for value in raw_binary):
                raise ValueError("Binary state is not 0/1")
            numeric = self.bundle["scaler"].transform(pd.DataFrame([raw_numeric], columns=names))[0].astype(np.float32)
            if not np.isfinite(numeric).all():
                raise ValueError("Transformed feature overflow")
            binary = np.array(raw_binary, dtype=np.int64)
            if self.last_time is not None:
                seconds = (timestamp - self.last_time).total_seconds()
                if seconds <= 0:
                    raise ValueError("Duplicate or out-of-order timestamp")
                if seconds != 1:
                    self.reset()
                    result["detail"] = "Time gap: history reset"

            # Predict t from strictly earlier observations; append x[t] only afterward.
            result["history_size"] = len(self.buffer)
            if len(self.buffer) < self.sequence_length:
                result["status"] = "WARMUP"
            else:
                history = torch.from_numpy(np.stack(self.buffer)).unsqueeze(0)
                numeric_target = torch.from_numpy(numeric).unsqueeze(0)
                binary_target = torch.from_numpy(binary).unsqueeze(0)
                with torch.inference_mode():
                    prediction, logits = self.model(history)
                    score = mixed_forecast_scores(prediction, logits, numeric_target, binary_target).item()
                    numeric_error = (prediction - numeric_target).square().mean().item()
                anomaly = int(score > self.threshold)
                result.update(status="ANOMALY" if anomaly else "NORMAL", prediction=anomaly,
                              score=score, numeric_mse=numeric_error, binary_ce=score-numeric_error)
            self.buffer.append(np.concatenate([numeric, binary]).astype(np.float32))
            self.last_time = timestamp
        except (KeyError, ValueError, TypeError, OverflowError) as error:
            self.reset()
            result.update(status="INSUFFICIENT_DATA", prediction=None, score=None,
                          numeric_mse=None, binary_ce=None, history_size=0, detail=str(error))
        result["processing_ms"] = (time.perf_counter() - started) * 1000
        return result


def simulate(model_dir, input_file, output_dir, speed=0., max_rows=None, start_row=0,
             print_every=100, quiet=False, mqtt_config=None):
    if not math.isfinite(speed) or speed < 0 or start_row < 0 or print_every < 1:
        raise ValueError("Invalid speed/start_row/print_every")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    model_dir, input_file, output_dir = map(Path, (model_dir, input_file, output_dir))
    detector = StreamingDetector.from_run(model_dir)
    if output_dir.exists():
        raise FileExistsError("Output directory exists; select a new simulation name")
    input_hash = file_hash(input_file)
    # Only headers and the current row are read, not the complete future series.
    with input_file.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=";")
        required = {"time", *detector.bundle["numeric_columns"], *detector.bundle["binary_columns"]}
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not required.issubset(reader.fieldnames):
            raise ValueError("Missing/duplicate input columns")
        output_dir.mkdir(parents=True, exist_ok=False)
        publisher = None
        write_json(output_dir / "run_config.json", {
            "model_dir": str(model_dir), "input_file": str(input_file), "input_sha256": input_hash,
            "model_sha256": file_hash(model_dir / "best_model.pt"),
            "threshold_file_sha256": file_hash(model_dir / "threshold.json"),
            "threshold": detector.threshold, "sequence_length": detector.sequence_length,
            "artifact_metadata": getattr(detector, "artifact_metadata", {}),
            "speed": speed, "start_row": start_row, "max_rows": max_rows,
            "ground_truth_used": False, "processing_ms_scope": "preprocessing + model + decision, excludes file logging and pacing",
            "mqtt": mqtt_config,
        })
        counts = {name: 0 for name in ("WARMUP", "NORMAL", "ANOMALY", "INSUFFICIENT_DATA")}
        count, state = 0, "completed"
        started = time.perf_counter()
        previous_timestamp = None
        with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as history_file, \
             (output_dir / "alerts.csv").open("w", encoding="utf-8", newline="") as alert_file:
            history_writer = csv.DictWriter(history_file, fieldnames=FIELDS)
            alert_writer = csv.DictWriter(alert_file, fieldnames=FIELDS)
            history_writer.writeheader()
            alert_writer.writeheader()
            history_file.flush()
            alert_file.flush()
            try:
                if mqtt_config is not None:
                    from src.mqtt_publisher import MqttPublisher
                    publisher = MqttPublisher(output_dir, model_run=model_dir.name, **mqtt_config)
                for source_index, row in enumerate(reader):
                    if source_index < start_row:
                        continue
                    if max_rows is not None and count >= max_rows:
                        break
                    # Labels and all unrelated fields are deliberately excluded.
                    observation = {name: row[name] for name in required}
                    try:
                        current_timestamp = datetime.strptime(observation["time"], "%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        current_timestamp = None
                    if speed and current_timestamp is not None and previous_timestamp is not None:
                        delay = (current_timestamp - previous_timestamp).total_seconds()
                        if delay == 1:
                            time.sleep(delay / speed)
                    previous_timestamp = current_timestamp
                    record = detector.push(observation)
                    count += 1
                    record["event_id"] = f"row-{source_index}"
                    counts[record["status"]] += 1
                    history_writer.writerow(record)
                    history_file.flush()
                    if record["status"] == "ANOMALY":
                        alert_writer.writerow(record)
                        alert_file.flush()
                        if not quiet:
                            print(f"WARNING {record['timestamp']} | ANOMALY | score={record['score']:.6f} "
                                  f"> threshold={detector.threshold:.6f}", flush=True)
                    elif not quiet and (count <= 3 or count % print_every == 0 or record["status"] == "INSUFFICIENT_DATA"):
                        score_text = "N/A" if record["score"] is None else f"{record['score']:.6f}"
                        print(f"{record['timestamp']} | {record['status']} | score={score_text} | {record['detail']}", flush=True)
                    # Local predictions/alerts are flushed before network publication.
                    if publisher is not None:
                        publisher.submit(record)
            except KeyboardInterrupt:
                state = "interrupted_by_user"
            except Exception:
                state = "failed"
                raise
            finally:
                if count == 0 and state == "completed":
                    state = "no_rows_processed"
                summary = {"status": state, "rows_processed": count, "counts": counts,
                           "threshold": detector.threshold, "elapsed_seconds": time.perf_counter()-started,
                           "trained_or_recalibrated": False}
                if publisher is not None:
                    summary["mqtt"] = publisher.close()
                write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="mixed_v2_full_seed42_run1")
    parser.add_argument("--input-file", type=Path, default=PROJECT_ROOT / "data/raw/hai/holdout/test2.csv")
    parser.add_argument("--simulation-name", default=datetime.now(timezone.utc).strftime("replay_%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--speed", type=float, default=0., help="0: no delay; 1: one second per observation; 10: ten times faster")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--start-row", type=int, default=0, help="Zero-based data row; a new warmup is always required")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-record console output, retain logs")
    parser.add_argument("--mqtt", action="store_true", help="Publish QoS 1 events and keep a local MQTT outbox")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic-prefix", default="industrial/anomaly")
    args = parser.parse_args()
    for name in (args.run_name, args.simulation_name):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            parser.error("Names must be single safe directory components")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    simulate(PROJECT_ROOT / "artifacts/hai/transformer_mixed" / args.run_name,
             args.input_file, PROJECT_ROOT / "results/hai/streaming" / args.simulation_name,
             args.speed, args.max_rows, args.start_row, args.print_every, args.quiet,
             {"host": args.mqtt_host, "port": args.mqtt_port,
              "topic_prefix": args.mqtt_topic_prefix} if args.mqtt else None)


if __name__ == "__main__":
    main()
