"""Non-blocking MQTT QoS 1 publisher backed by a local append-only outbox."""
import json
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class MqttPublisher:
    def __init__(self, output_dir, host="127.0.0.1", port=1883,
                 topic_prefix="industrial/anomaly", model_run="unknown",
                 simulation_id=None, client_factory=None):
        if not 1 <= port <= 65535 or not host or any(c in topic_prefix for c in "+#\x00"):
            raise ValueError("Invalid MQTT host, port or topic prefix")
        self.prefix = topic_prefix.strip("/")
        if not self.prefix:
            raise ValueError("Topic prefix cannot be empty")
        self.host, self.port = host, port
        self.model_run = model_run
        self.simulation_id = simulation_id or uuid.uuid4().hex
        self.connected = threading.Event()
        self.network_events = queue.SimpleQueue()
        self.pending = {}
        self.queued, self.sent, self.acknowledged = 0, 0, 0
        self.last_error = None
        self.closed = False
        output_dir = Path(output_dir)
        self.outbox_path = output_dir / "mqtt_outbox.jsonl"
        self.delivery_path = output_dir / "mqtt_delivery.jsonl"
        # Exclusive creation prevents accidental loss of earlier unsent events.
        self.outbox = self.outbox_path.open("x", encoding="utf-8")
        self.reader = self.outbox_path.open("r", encoding="utf-8")
        self.delivery = self.delivery_path.open("x", encoding="utf-8")
        factory = client_factory or mqtt.Client
        self.client = factory(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                              client_id=f"hai-{self.simulation_id[:20]}", protocol=mqtt.MQTTv311)
        self.client.connect_timeout = 1.0
        self.client.max_queued_messages_set(256)
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_connect_fail = self._on_connect_fail
        try:
            self.client.connect_async(host, port, keepalive=30)
            self.client.loop_start()
        except (OSError, ValueError, RuntimeError) as error:
            self._log("connection_error", detail=str(error))

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.connected.set()
            self.network_events.put(("connected", str(reason_code)))
        else:
            self.connected.clear()
            self.network_events.put(("connection_rejected", str(reason_code)))

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected.clear()
        self.network_events.put(("disconnected", str(reason_code)))

    def _on_connect_fail(self, client, userdata):
        self.connected.clear()
        self.network_events.put(("connection_error", "Broker unavailable; events retained in local outbox"))

    def _log(self, status, **fields):
        self.delivery.write(json.dumps(dict(timestamp_utc=utc_now(), status=status, **fields), allow_nan=False) + "\n")
        self.delivery.flush()
        if status in ("connection_error", "connection_rejected", "publish_error"):
            self.last_error = fields.get("detail", status)

    def submit(self, record):
        for kind in (["predictions", "alerts"] if record["status"] == "ANOMALY" else ["predictions"]):
            event_id = f"{self.simulation_id}:{record['event_id']}:{kind}"
            payload = dict(record, event_id=event_id, source_event_id=record["event_id"],
                           simulation_id=self.simulation_id, message_type=kind,
                           model_run=self.model_run, schema_version=1, created_at_utc=utc_now())
            self.enqueue({"event_id": event_id, "topic": f"{self.prefix}/{kind}", "payload": payload})

    def enqueue(self, envelope):
        if self.closed:
            raise RuntimeError("Publisher is closed")
        self.outbox.write(json.dumps(envelope, allow_nan=False) + "\n")
        self.outbox.flush()
        self.queued += 1
        self.poll()

    def poll(self):
        # Only the simulation thread touches files and the pending map.
        while not self.network_events.empty():
            status, detail = self.network_events.get()
            self._log(status, detail=detail)
        for mid, (info, envelope) in list(self.pending.items()):
            if info.is_published():
                self._log("broker_ack", event_id=envelope["event_id"], topic=envelope["topic"], qos=1)
                self.acknowledged += 1
                del self.pending[mid]
        if not self.connected.is_set():
            return
        while len(self.pending) < 128:
            position = self.reader.tell()
            line = self.reader.readline()
            if not line:
                break
            envelope = json.loads(line)
            try:
                info = self.client.publish(envelope["topic"],
                                           json.dumps(envelope["payload"], allow_nan=False),
                                           qos=1, retain=False)
            except (OSError, ValueError, RuntimeError) as error:
                self.reader.seek(position)
                self._log("publish_error", detail=str(error), event_id=envelope["event_id"])
                break
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                self.reader.seek(position)
                self._log("publish_error", detail=f"Paho return code {info.rc}", event_id=envelope["event_id"])
                break
            self.pending[info.mid] = (info, envelope)
            self.sent += 1

    def close(self, timeout=3.):
        if self.closed:
            return self.summary
        deadline = time.monotonic() + max(0., timeout)
        while self.acknowledged < self.queued and time.monotonic() < deadline:
            self.poll()
            time.sleep(.02)
        self.poll()
        self.client.disconnect()
        self.client.loop_stop()
        self.summary = {
            "enabled": True, "host": self.host, "port": self.port, "topic_prefix": self.prefix,
            "simulation_id": self.simulation_id, "qos": 1, "retain": False,
            "queued": self.queued, "sent": self.sent, "broker_acknowledged": self.acknowledged,
            "unconfirmed": self.queued - self.acknowledged, "last_error": self.last_error,
            "ack_scope": "broker PUBACK, not subscriber receipt",
        }
        self._log("publisher_closed", **self.summary)
        self.closed = True
        self.reader.close()
        self.outbox.close()
        self.delivery.close()
        return self.summary
