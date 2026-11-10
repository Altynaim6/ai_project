"""Subscribe to lab events; log each receipt and flag duplicate event IDs."""
import argparse
import json
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from src.mqtt_publisher import utc_now


def subscribe(output, host="127.0.0.1", port=1883, topic="industrial/anomaly/#",
              timeout=60., max_messages=0, ready_file=None, quiet=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ready = threading.Event()
    done = threading.Event()
    seen, counts, errors = set(), {}, []
    stats = {"received": 0, "unique": 0, "duplicates": 0, "invalid": 0}
    stream = output.open("x", encoding="utf-8")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv311)
    client.connect_timeout = 2.

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            errors.append(f"Connection rejected: {reason_code}")
            done.set()
        else:
            result, _ = client.subscribe(topic, qos=1)
            if result != mqtt.MQTT_ERR_SUCCESS:
                errors.append(f"Subscribe failed: {result}")
                done.set()

    def on_subscribe(client, userdata, mid, reason_codes, properties):
        if any(code.is_failure for code in reason_codes):
            errors.append("Broker rejected subscription")
            done.set()
            return
        ready.set()
        if ready_file:
            Path(ready_file).write_text(json.dumps({"status": "subscribed", "topic": topic}), encoding="utf-8")
        if not quiet:
            print(f"SUBSCRIBED {topic}", flush=True)

    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            event_id = payload["event_id"]
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("Invalid event_id")
            duplicate = event_id in seen
            seen.add(event_id)
            stats["received"] += 1
            stats["duplicates"] += int(duplicate)
            stats["unique"] += int(not duplicate)
            counts[message.topic] = counts.get(message.topic, 0) + 1
            stream.write(json.dumps({"received_at_utc": utc_now(), "topic": message.topic,
                                     "qos": message.qos, "duplicate": duplicate, "payload": payload}) + "\n")
            stream.flush()
            if not quiet:
                print(f"{message.topic} | {event_id} | {payload.get('status')} | duplicate={duplicate}", flush=True)
            if max_messages and stats["received"] >= max_messages:
                done.set()
        except (UnicodeError, ValueError, KeyError, TypeError) as error:
            stats["invalid"] += 1
            stream.write(json.dumps({"received_at_utc": utc_now(), "topic": message.topic,
                                     "error": str(error)}) + "\n")
            stream.flush()

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    try:
        client.connect(host, port, keepalive=30)
        client.loop_start()
        if not ready.wait(5):
            raise RuntimeError("Subscription was not acknowledged within 5 seconds")
        deadline = time.monotonic() + timeout if timeout else float("inf")
        while not done.wait(.1) and time.monotonic() < deadline:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()
        client.loop_stop()
        stream.close()
    stats.update(topics=counts, errors=errors, subscribed=ready.is_set())
    output.with_suffix(".summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="industrial/anomaly/#")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--max-messages", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.timeout < 0 or args.max_messages < 0:
        parser.error("Timeout and max-messages must be non-negative")
    subscribe(args.output, args.host, args.port, args.topic, args.timeout,
              args.max_messages, args.ready_file, args.quiet)


if __name__ == "__main__":
    main()
