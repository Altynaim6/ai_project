import json
from types import SimpleNamespace

from src.mqtt_publisher import MqttPublisher
from src.mqtt_replay import replay
from src.verify_mqtt_delivery import verify_delivery


class FakeClient:
    online = True
    sent_messages = []

    def __init__(self, **kwargs):
        self.mid = 0
    def max_queued_messages_set(self, value):
        pass
    def reconnect_delay_set(self, **kwargs):
        pass
    def connect_async(self, *args, **kwargs):
        pass
    def loop_start(self):
        if self.online:
            self.on_connect(self, None, None, 0, None)
        else:
            self.on_connect_fail(self, None)
    def publish(self, topic, payload, qos, retain):
        assert qos == 1 and retain is False
        self.mid += 1
        self.sent_messages.append((topic, json.loads(payload)))
        return SimpleNamespace(rc=0, mid=self.mid, is_published=lambda: True)
    def disconnect(self):
        self.on_disconnect(self, None, None, 0, None)
    def loop_stop(self):
        pass


def record(index, status="NORMAL"):
    return {"event_id": f"row-{index}", "timestamp": "2020-01-01 00:00:01",
            "status": status, "prediction": int(status == "ANOMALY"), "score": .2,
            "threshold": .1}


def test_qos_topics_unique_ids_and_broker_ack(tmp_path):
    FakeClient.sent_messages = []
    publisher = MqttPublisher(tmp_path, simulation_id="fixed-run", client_factory=FakeClient)
    publisher.submit(record(0, "WARMUP"))
    publisher.submit(record(1, "NORMAL"))
    publisher.submit(record(2, "ANOMALY"))
    summary = publisher.close(.1)
    assert summary["queued"] == summary["broker_acknowledged"] == 4
    assert summary["unconfirmed"] == 0
    messages = FakeClient.sent_messages
    assert [topic for topic, _ in messages].count("industrial/anomaly/predictions") == 3
    assert [topic for topic, _ in messages].count("industrial/anomaly/alerts") == 1
    assert len({payload["event_id"] for _, payload in messages}) == 4
    outbox = [json.loads(line) for line in (tmp_path / "mqtt_outbox.jsonl").read_text().splitlines()]
    assert len(outbox) == 4
    assert outbox[-1]["payload"]["source_event_id"] == "row-2"


def test_outage_retains_events_and_live_reconnect_drains(tmp_path):
    class OfflineClient(FakeClient):
        online = False
    publisher = MqttPublisher(tmp_path, client_factory=OfflineClient)
    publisher.submit(record(0))
    publisher.submit(record(1, "ANOMALY"))
    assert publisher.queued == 3 and not publisher.pending
    assert publisher.acknowledged == 0
    publisher._on_connect(publisher.client, None, None, 0, None)
    publisher.poll()
    summary = publisher.close(.1)
    assert summary["broker_acknowledged"] == 3


def test_outage_replay_preserves_ids_and_skips_confirmed(tmp_path, monkeypatch):
    class OfflineClient(FakeClient):
        online = False
    publisher = MqttPublisher(tmp_path, simulation_id="original-run", client_factory=OfflineClient)
    publisher.submit(record(0, "ANOMALY"))
    summary = publisher.close(0)
    assert summary["unconfirmed"] == 2
    original = [json.loads(line) for line in (tmp_path / "mqtt_outbox.jsonl").read_text().splitlines()]
    monkeypatch.setattr("src.mqtt_publisher.mqtt.Client", FakeClient)
    FakeClient.sent_messages = []
    result = replay(tmp_path, timeout=.1)
    assert result["broker_acknowledged"] == 2
    assert {m[1]["event_id"] for m in FakeClient.sent_messages} == {m["event_id"] for m in original}
    result = replay(tmp_path, timeout=.1)
    assert result["queued"] == 0


def test_delivery_verifier_accounts_for_duplicate_receipts(tmp_path):
    publisher = MqttPublisher(tmp_path, client_factory=FakeClient)
    publisher.submit(record(1, "ANOMALY"))
    publisher.close(.1)
    events = [json.loads(line) for line in (tmp_path / "mqtt_outbox.jsonl").read_text().splitlines()]
    receipts = [{"topic": event["topic"], "payload": event["payload"], "qos": 1} for event in events]
    received = tmp_path / "received.jsonl"
    received.write_text("\n".join(json.dumps(event) for event in receipts + receipts[:1]))
    result = verify_delivery(tmp_path, received)
    assert result["passed"] and result["duplicate_deliveries"] == 1
