# MQTT и Mosquitto для потоковой симуляции v2

Все команды ниже выполняются в PowerShell из корня `industrial-anomaly-detection`. Для демонстрации нужны три терминала VS Code: брокер, подписчик и симулятор.

## 1. Подготовка

В окружении проекта уже установлен `paho-mqtt==2.2.0`. Проверка:

```powershell
.\.venv\Scripts\python.exe -c "import paho.mqtt.client as mqtt; print(mqtt.CallbackAPIVersion.VERSION2)"
```

Переносимый Mosquitto 2.1.2 извлечён в `tools/mosquitto/` вместе с DLL, README и лицензиями. Служба Windows не устанавливалась. `tools/` с бинарными файлами исключён из Git. Если переносите только исходный код на другой компьютер, получите Mosquitto из [официального источника](https://mosquitto.org/download/) и используйте путь к установленному `mosquitto.exe` вместо локального пути в командах.

## 2. Терминал 1: запустить брокер

```powershell
.\tools\mosquitto\mosquitto.exe -c .\configs\mosquitto-local.conf -v
```

Оставьте терминал работающим. Конфигурация слушает только `127.0.0.1:1883`, разрешает анонимные локальные подключения и пишет лог в консоль. Это лабораторная конфигурация для одного компьютера. Если порт занят уже запущенным брокером, используйте существующий подходящий брокер либо выберите другой порт в конфиге и одинаковый порт у клиентов.

## 3. Терминал 2: запустить подписчика

Для первой демонстрации:

```powershell
.\.venv\Scripts\python.exe -m src.mqtt_subscriber --output results\mqtt_manual\received.jsonl --max-messages 162 --timeout 120
```

Дождитесь `SUBSCRIBED industrial/anomaly/#`, только затем запускайте симулятор. Подписчик завершится после 162 сообщений или таймаута. Для следующей демонстрации задайте новое имя выходного файла: существующий лог не перезаписывается.

Для непрерывного просмотра до Ctrl+C:

```powershell
.\.venv\Scripts\python.exe -m src.mqtt_subscriber --output results\mqtt_manual\monitor.jsonl --timeout 0
```

Альтернатива — штатный клиент Mosquitto (без нашего JSON-лога и дедупликации):

```powershell
.\tools\mosquitto\mosquitto_sub.exe -h 127.0.0.1 -p 1883 -t "industrial/anomaly/#" -q 1 -v
```

## 4. Терминал 3: запустить симуляцию

```powershell
.\.venv\Scripts\python.exe -m src.simulate_transformer_mixed --mqtt --start-row 176 --max-rows 160 --speed 10 --print-every 30 --simulation-name mqtt_manual_01
```

Этот участок ранее оценённого test2 выбран для короткой демонстрации предупреждений, а не для повторной оценки качества. Ожидаются 30 WARMUP, 128 NORMAL и 2 ANOMALY. Всего MQTT-сообщений — 162: 160 результатов обработки (включая warmup) и 2 предупреждения. Модель, preprocessing и порог 0.1558542103 загружаются из сохранённого запуска; обучение и новая калибровка не выполняются.

Для работы без задержек используйте `--speed 0`; `--speed 1` приблизительно воспроизводит секундные интервалы. При запуске всей записи это длительный процесс, поэтому для демонстрации оставляйте max-rows.

Параметры MQTT:

```text
--mqtt                         включить публикацию
--mqtt-host 127.0.0.1           адрес брокера
--mqtt-port 1883                порт
--mqtt-topic-prefix industrial/anomaly
```

Без `--mqtt` симулятор работает в прежнем локальном режиме.

## 5. Что публикуется

| Topic | События |
|---|---|
| `industrial/anomaly/predictions` | Каждый обработанный результат: WARMUP, NORMAL, ANOMALY, INSUFFICIENT_DATA |
| `industrial/anomaly/alerts` | Дополнительное событие для каждого ANOMALY |

QoS=1, retain=false. JSON содержит schema_version, уникальный event_id, source_event_id, simulation_id, message_type, model_run, исходный timestamp, status, prediction, score, threshold, компоненты ошибки и created_at_utc. У warmup/ошибок prediction и score равны JSON null. Метки attack в сообщение не включаются.

Пример сокращённого предупреждения:

```json
{
  "schema_version": 1,
  "event_id": "<simulation-id>:row-236:alerts",
  "source_event_id": "row-236",
  "simulation_id": "<simulation-id>",
  "message_type": "alerts",
  "model_run": "mixed_v2_full_seed42_run1",
  "timestamp": "2019-11-04 15:03:56",
  "status": "ANOMALY",
  "prediction": 1,
  "score": 1.740708,
  "threshold": 0.1558542102575302
}
```

Предупреждение означает превышение порога модели, не подтверждённую атаку. Predictions и alerts одной исходной строки имеют разные event_id, поскольку это два разных сообщения; source_event_id у них общий.

## 6. Проверить доставку

После завершения подписчика и симуляции:

```powershell
.\.venv\Scripts\python.exe -m src.verify_mqtt_delivery results\hai\streaming\mqtt_manual_01 results\mqtt_manual\received.jsonl
```

Ожидаются passed=true, expected=162, received_unique=162, missing=0, unconfirmed_by_broker=0. Проверяются QoS, topics, все поля payload и исходные event_id; дубликаты считаются отдельно. Дополнительные чужие сообщения в файле приводят к отказу проверки — используйте отдельный лог/префикс для демонстрации.

Само поле `broker_acknowledged` подтверждает PUBACK от брокера, а не получение подписчиком. Наш verifier дополнительно сверяет журнал подписчика. QoS 1 допускает повторную доставку; подписчик помечает одинаковые event_id как duplicate=true и отдельно считает unique. Память дедупликации действует в пределах одного запуска подписчика, а JSON-журнал позволяет анализировать повторные доставки позднее.

## 7. Недоступный брокер и восстановление

Проверка на заведомо свободном порту, на котором брокер не запущен:

```powershell
.\.venv\Scripts\python.exe -m src.simulate_transformer_mixed --mqtt --mqtt-port 18884 --start-row 176 --max-rows 160 --speed 0 --simulation-name mqtt_offline_01
```

Предсказания и локальные предупреждения продолжатся. В конце ожидаются mqtt.queued=162, mqtt.broker_acknowledged=0, mqtt.unconfirmed=162. Ошибка соединения записана в mqtt_delivery.jsonl, а все события находятся в mqtt_outbox.jsonl. Указанный порт должен действительно быть свободен, иначе это не тест отказа.

После запуска брокера на 1883 откройте новый подписчик:

```powershell
.\.venv\Scripts\python.exe -m src.mqtt_subscriber --output results\mqtt_manual\recovered.jsonl --max-messages 162 --timeout 120
```

Дождитесь SUBSCRIBED, затем в другом терминале:

```powershell
.\.venv\Scripts\python.exe -m src.mqtt_replay results\hai\streaming\mqtt_offline_01 --host 127.0.0.1 --port 1883
```

Повторно отправляются только события без подтверждения в исходном журнале и журналах прошлых replay. Topics, payload, event_id и время создания сохраняются. Повторный запуск после успешного replay должен показать queued=0. Если подтверждение потерялось, сообщение может прийти повторно — получатель должен учитывать event_id.

```powershell
.\.venv\Scripts\python.exe -m src.verify_mqtt_delivery results\hai\streaming\mqtt_offline_01 results\mqtt_manual\recovered.jsonl
```

Если соединение возвращается во время самой симуляции, клиент автоматически переподключается и продолжает чтение локального outbox. Публикация выполняется через сетевой поток Paho; локальные CSV сбрасываются до MQTT-отправки. При завершении ожидание подтверждений ограничено тремя секундами; неподтверждённые события остаются на диске для ручного replay. Replay по умолчанию ждёт подтверждения до пяти секунд, это можно увеличить через `--timeout`.

## 8. Журналы

В каталоге симуляции:

- `predictions.csv`, `alerts.csv` — исходные локальные результаты.
- `mqtt_outbox.jsonl` — неизменяемые исходящие события, записанные до отправки.
- `mqtt_delivery.jsonl` — соединения, ошибки и broker_ack для каждого подтверждённого event_id.
- `summary.json` — счётчики MQTT; unconfirmed не трактуется как доказанная потеря, поскольку сообщение могло дойти при потере ACK.
- `mqtt_verification.json` — результат сверки с подписчиком.
- `mqtt_replay_<UTC>/` — отдельные журналы повторной отправки.

JSONL и CSV используют flush после записи; это не обещание сохранности при внезапном отключении питания. processing_ms в predictions измеряет preprocessing + модель + решение, исключает MQTT, запись файлов и ожидание между записями; он не заменяет end-to-end benchmark.

## 9. Завершение и источники

Ctrl+C останавливает монитор/симулятор. В терминале брокера Ctrl+C завершает локальный Mosquitto. Временный брокер, использованный при разработке, останавливается после проверок; для своей демонстрации запустите терминал 1.

Использованы [Paho Python API](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html), [переход на callback API v2](https://eclipse.dev/paho/files/paho.mqtt.python/html/migrations.html), [конфигурация Mosquitto](https://mosquitto.org/man/mosquitto-conf-5.html). Для будущего подключения с другого компьютера потребуется отдельная конфигурация адресов, аутентификации и доступа; текущая проверенная конфигурация остаётся локальной.
