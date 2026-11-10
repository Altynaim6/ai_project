"""Compare saved outbox, broker acknowledgements and subscriber receipts."""
import argparse
import json
from collections import Counter
from pathlib import Path


def verify_delivery(simulation_dir, received_file):
    simulation_dir = Path(simulation_dir)
    expected_rows = [json.loads(line) for line in (simulation_dir / "mqtt_outbox.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {row["event_id"]: row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Duplicate event IDs in original outbox")
    received = [json.loads(line) for line in Path(received_file).read_text(encoding="utf-8").splitlines()]
    unique = {}
    duplicates = 0
    for row in received:
        payload = row.get("payload", {})
        event_id = payload.get("event_id")
        if event_id not in expected:
            raise ValueError("Unexpected or invalid subscriber event")
        if row["qos"] != 1 or row["topic"] != expected[event_id]["topic"] or payload != expected[event_id]["payload"]:
            raise ValueError("Received topic/QoS/payload differs from the original event")
        duplicates += int(event_id in unique)
        unique[event_id] = row
    missing = set(expected) - set(unique)
    acked = set()
    for path in [simulation_dir / "mqtt_delivery.jsonl", *simulation_dir.glob("mqtt_replay_*/mqtt_delivery.jsonl")]:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["status"] == "broker_ack":
                acked.add(record["event_id"])
    unconfirmed = set(expected) - acked
    result = {"passed": not missing and not unconfirmed, "expected": len(expected),
              "received_unique": len(unique), "duplicate_deliveries": duplicates,
              "missing": len(missing), "unconfirmed_by_broker": len(unconfirmed),
              "topics": dict(Counter(row["topic"] for row in unique.values())),
              "payloads_match_original": True, "received_qos": 1}
    print(json.dumps(result, indent=2))
    (simulation_dir / "mqtt_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError("MQTT delivery verification failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_dir", type=Path)
    parser.add_argument("received_file", type=Path)
    args = parser.parse_args()
    verify_delivery(args.simulation_dir, args.received_file)
