"""Replay events lacking a broker ACK, retaining original topics and event IDs."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.mqtt_publisher import MqttPublisher


def replay(simulation_dir, host="127.0.0.1", port=1883, timeout=5.):
    simulation_dir = Path(simulation_dir)
    acknowledged = set()
    journals = [simulation_dir / "mqtt_delivery.jsonl", *simulation_dir.glob("mqtt_replay_*/mqtt_delivery.jsonl")]
    for journal in journals:
        if journal.exists():
            for line in journal.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["status"] == "broker_ack":
                    acknowledged.add(record["event_id"])
    outbox = simulation_dir / "mqtt_outbox.jsonl"
    if not outbox.is_file():
        raise FileNotFoundError(outbox)
    output = simulation_dir / datetime.now(timezone.utc).strftime("mqtt_replay_%Y%m%dT%H%M%S%fZ")
    output.mkdir()
    publisher = MqttPublisher(output, host=host, port=port)
    try:
        with outbox.open(encoding="utf-8") as source:
            for line in source:
                envelope = json.loads(line)
                if envelope["event_id"] not in acknowledged:
                    publisher.enqueue(envelope)
    finally:
        summary = publisher.close(timeout)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved:", output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--timeout", type=float, default=5.)
    args = parser.parse_args()
    if args.timeout < 0:
        parser.error("timeout must be non-negative")
    replay(args.simulation_dir, args.host, args.port, args.timeout)


if __name__ == "__main__":
    main()
