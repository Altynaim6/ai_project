# Mixed Transformer: подготовленный конвейер

Запускать из корня `industrial-anomaly-detection`:

```powershell
.\.venv\Scripts\python.exe -m src.preprocess_hai_mixed
.\.venv\Scripts\python.exe -m src.check_transformer_mixed
```

Предобработка читает существующие `data/processed/hai/train.csv` и `validation.csv`, проверяет схему, полные данные, нормальные метки, непрерывность времени и отсутствие перекрытия. Test не читается. Числовой StandardScaler fit выполняется только на train. Четыре Boolean-канала остаются 0/1, даже когда постоянны на train.

Результаты: `data/processed/hai_mixed/{train,validation}.npz` и `artifacts/hai/preprocessing_mixed/{preprocessing.joblib,manifest.json}`. Manifest хранит порядок входов/выходов, интервалы, хэши источников и схемы. Повторный запуск заменяет только эти mixed-файлы. Старые numeric smoke-артефакты не меняются.

Контракт `MixedForecastWindowDataset`: история `[B,L,59]` в порядке 55 числовых + 4 булевых; цель numeric `[B,55]` float32 и цель binary `[B,4]` int64. История не содержит целевой строки. Окна строятся внутри каждого split и непрерывного сегмента: положительный разрыв времени начинает новый сегмент. Дубликаты и обратный порядок времени отклоняются. Индексы оцениваемых целей доступны через `target_indices`.

`MixedForecastTransformer` возвращает numeric prediction `[B,55]` и raw binary logits `[B,4,2]`. Обе головы прогнозируют следующий шаг. Не применять softmax до CrossEntropy.

Пример будущего шага обучения:

```python
history, numeric_target, binary_target = batch
optimizer.zero_grad(set_to_none=True)
numeric_prediction, binary_logits = model(history)
scores = mixed_forecast_scores(
    numeric_prediction, binary_logits, numeric_target, binary_target
)
loss = scores.mean()
loss.backward()
optimizer.step()
```

Для inference применять сохранённый bundle, не вызывать fit. В checkpoint сохранять `model_config`, state dict, списки numeric/binary каналов и schema hash, затем связывать его с конкретным bundle/manifest. Проверочный скрипт временно сохраняет не обученную модель и проверяет восстановление; его loss не является результатом обучения.

## Обучение, лучшая эпоха и порог

Полное обучение (параметры по умолчанию: максимум 50 эпох, patience 5, seed 42):

```powershell
.\.venv\Scripts\python.exe -m src.train_transformer_mixed --run-name mixed_full_seed42
```

Все train-окна используются на каждой эпохе, в перемешанном порядке. Validation всегда содержит все 61890 окон, включая неполный последний batch; перемешивание и dropout при оценке отключены. Среднее loss взвешено по числу записей, а не по числу пакетов. В историю отдельно записываются MSE, CE, суммарная ошибка, число окон и время.

Строгое снижение полной validation loss сохраняет `best_model.pt`; после пяти эпох без улучшения срабатывает early stopping. В конце заново загружается именно лучший checkpoint. Полная validation оценивается ещё раз, результат сверяется с выбранной эпохой и рассчитывается 99-й процентиль score (NumPy linear quantile). Решение — строго `score > threshold`. Доля превышений около 1% обусловлена калибровкой; это не независимая оценка FPR на test. Эта validation используется и для выбора эпохи, и для порога, что явно раскрывается в артефактах.

Короткая проверка инфраструктуры, с полной validation, но ограниченным train:

```powershell
.\.venv\Scripts\python.exe -m src.train_transformer_mixed --epochs 1 --max-train-batches 2 --run-name mixed_smoke_check
```

Такой запуск получает `run_type=smoke_test`. Его порог не является итоговым результатом. Чтобы начать полное обучение, не передавать `--max-train-batches`.

Каждый запуск получает отдельные каталоги:

- `artifacts/hai/transformer_mixed/<run-name>/`: лучший checkpoint, копия preprocessing, manifest предобработки, run config и новый `threshold.json`.
- `results/hai/transformer_mixed/<run-name>/`: `history.csv/json`, `validation_scores.csv`, `summary.json`.

Уже существующее имя запуска отклоняется без перезаписи. Если `--run-name` не задан, создаётся уникальная UTC-метка. Конфигурация сохраняет версии, seed, устройство, потоки и хэши исходных NPZ/preprocessing/схемы; порог связан с хэшем конкретного checkpoint. `summary.json` со статусом completed появляется только после успешной калибровки. При прерывании история и лучший завершённый checkpoint сохраняются, но автоматическое продолжение обучения в этой версии не реализовано.

Доступны `--epochs`, `--patience`, `--batch-size`, `--validation-batch-size`, `--seed`, `--threads` (по умолчанию 4), `--device cpu|cuda`. CUDA требует соответствующей установки PyTorch. На CPU полное обучение может занять значительное время; прогресс печатается каждые 100 пакетов. После запуска не менять входные данные или preprocessing.

Старый `train_transformer.py` относится к однотипному numeric baseline, запускать его для mixed-модели нельзя. Старый порог 0.5434 неприменим к сумме MSE+CE. Test не читается ни при обучении, ни при выборе checkpoint, ни при калибровке.

## Оценка на test1.csv

После завершения полного обучения и появления `threshold.json` выполнить:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_transformer_mixed --run-name mixed_full_seed42
```

Скрипт использует `best_model.pt`, `preprocessing.joblib` и `threshold.json` из указанного запуска. Хэши связывают модель, preprocessing и порог; несовпадения отклоняются. Smoke-test модели не принимаются. Test не используется для fit, выбора эпохи или изменения порога. Указывая другую модель, использовать её собственное имя запуска; уже существующая папка оценки не перезаписывается.

История каждого предсказания содержит только L предыдущих записей. Цель и эталонная attack-метка относятся к следующему наблюдению. Первые L строк — warmup, они исключаются. Для текущего test1 при L=30 ожидаются 291570 оцениваемых целей из 291600 исходных строк. Все четыре attack-колонки исключены из входов. Непрерывность времени, отсутствие перекрытия с validation, булевы значения и схема проверяются до inference.

Результаты в `results/hai/transformer_mixed/<run-name>/test_evaluation/`:

- `metrics.json/csv`: Accuracy, Precision, Recall, F1, FPR, average precision, TP/FP/TN/FN, размеры выборки и хэши артефактов.
- `test_predictions.csv`: timestamp, true label, score, числовая/булева части ошибки, сохранённый порог и предсказание.
- `figures/confusion_matrix.png`, `scores_and_attacks.png`, `precision_recall.png`, `metrics.png`.
- `evaluation_status.json`: статус completed только после успешного завершения экспорта.

Метрики поточечные без point adjustment. PR-кривая описывает ранжирование, не используется для выбора нового порога. PR-AUC в виде average precision (AP) отделена от трапециевидной площади. Длительность evaluation loop включает пакетную обработку и не заменяет измерение inference latency batch=1.

После просмотра test-результатов не подбирать порог/архитектуру по этому test в рамках того же эксперимента; дальнейший подбор требует отдельного обозначенного протокола. Оценка HAI не является оценкой на SWaT и сама по себе не доказывает обнаружение zero-day атак.

Проверки `tests/test_hai_mixed.py`: train-only fit на данных с намеренным сдвигом validation, сохранение binary без масштабирования, восстановление scaler, корректная цель окна, запрет неверных состояний и перекрытий, эталонная формула loss, отсутствие внимания к будущему и градиенты обеих голов.
